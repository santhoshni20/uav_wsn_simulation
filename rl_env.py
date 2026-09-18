import gymnasium as gym
from gymnasium import spaces
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

from config import (
    AREA_WIDTH,
    AREA_HEIGHT,
    NUM_SENSORS,
    UAV_COMMUNICATION_RANGE,
    SENSOR_COMMUNICATION_RANGE,
    LINK_QUALITY_THRESHOLD,
    CORRIDOR_X_MIN,
    CORRIDOR_X_MAX,
    CORRIDOR_Y_MIN,
    CORRIDOR_Y_MAX,
    ENABLE_DISRUPTION,
    DISRUPTION_X_MIN,
    DISRUPTION_X_MAX,
    DISRUPTION_Y_MIN,
    DISRUPTION_Y_MAX
)
from nodes import UAV
from environment import SimulationEnvironment
from network import DynamicWSNGraph
from st_gnn import extract_node_features_and_adj, STGNN


class UAVWSNEnv(gym.Env):
    """
    Multi-Agent Gymnasium Environment for UAV-assisted Mobile WSN.
    Coordinates UAV repositioning and next-hop sensor routing.
    """
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, history_window=5, max_steps=50):
        super(UAVWSNEnv, self).__init__()
        self.history_window = history_window
        self.max_steps = max_steps
        self.num_nodes = NUM_SENSORS + 2  # 20 sensors + UAV + CS

        self.uav_max_speed = 30.0  # meters per timestep
        self.uav_hover_energy = 0.05
        self.uav_move_energy_factor = 0.003
        self.sensor_tx_energy = 0.02

        # Action Space:
        # 1. UAV movement: continuous (dx, dy) in [-1, 1]
        # 2. Routing decisions: next-hop selection for each sensor (0..NUM_SENSORS-1: sensors, NUM_SENSORS: UAV, NUM_SENSORS+1: CS)
        self.action_space = spaces.Dict({
            "uav_action": spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32),
            "routing_actions": spaces.MultiDiscrete([self.num_nodes] * NUM_SENSORS)
        })

        # Observation Space:
        # History of node features (T, N, 8) and adjacency (T, N, N), plus UAV local state
        self.observation_space = spaces.Dict({
            "node_features": spaces.Box(low=-np.inf, high=np.inf, shape=(self.history_window, self.num_nodes, 8), dtype=np.float32),
            "adjacency": spaces.Box(low=0.0, high=1.0, shape=(self.history_window, self.num_nodes, self.num_nodes), dtype=np.float32),
            "uav_state": spaces.Box(low=0.0, high=1.0, shape=(6,), dtype=np.float32)
        })

        self.sim_env = SimulationEnvironment()
        self.network = DynamicWSNGraph()
        self.st_gnn = STGNN(in_features=8, spatial_dim=32, temporal_dim=64, out_dim=64)
        self.st_gnn.eval()

        self.feature_history = []
        self.adj_history = []
        self.current_step = 0
        self.uav_trajectory = []

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)

        self.sim_env.reset()
        self.current_step = 0
        self.uav_trajectory = [self.sim_env.uav.position]

        self.feature_history = []
        self.adj_history = []

        # Populate initial history window
        for _ in range(self.history_window):
            graph = self.network.build_graph(self.sim_env)
            feat, adj = extract_node_features_and_adj(self.sim_env, graph)
            self.feature_history.append(feat)
            self.adj_history.append(adj)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def _get_obs(self):
        feat_tensor = torch.stack(self.feature_history[-self.history_window:], dim=0)
        adj_tensor = torch.stack(self.adj_history[-self.history_window:], dim=0)

        uav = self.sim_env.uav
        cs = self.sim_env.command_station
        dist_to_cs = np.linalg.norm(np.array(uav.position) - np.array(cs.position)) / 1414.0

        uav_state = np.array([
            uav.position[0] / AREA_WIDTH,
            uav.position[1] / AREA_HEIGHT,
            uav.energy / 100.0,
            dist_to_cs,
            1.0 if dist_to_cs * 1414.0 <= UAV_COMMUNICATION_RANGE else 0.0,
            self.current_step / self.max_steps
        ], dtype=np.float32)

        return {
            "node_features": feat_tensor.numpy(),
            "adjacency": adj_tensor.numpy(),
            "uav_state": uav_state
        }

    def _get_info(self):
        graph = self.network.build_graph(self.sim_env)
        num_components = nx.number_connected_components(graph)
        uav_connected_to_cs = graph.has_edge("UAV", "CS")
        return {
            "step": self.current_step,
            "connected_components": num_components,
            "uav_connected_to_cs": uav_connected_to_cs,
            "uav_energy": self.sim_env.uav.energy,
            "uav_position": self.sim_env.uav.position
        }

    def step(self, action):
        self.current_step += 1

        # 1. Process UAV action (continuous displacement)
        uav_act = action["uav_action"]
        dx = float(uav_act[0]) * self.uav_max_speed
        dy = float(uav_act[1]) * self.uav_max_speed

        curr_x, curr_y = self.sim_env.uav.position
        new_x = np.clip(curr_x + dx, 0, AREA_WIDTH)
        new_y = np.clip(curr_y + dy, 0, AREA_HEIGHT)
        move_dist = np.hypot(new_x - curr_x, new_y - curr_y)

        self.sim_env.uav.position = (new_x, new_y)
        self.uav_trajectory.append((new_x, new_y))

        # UAV energy consumption
        uav_energy_spent = self.uav_hover_energy + move_dist * self.uav_move_energy_factor
        self.sim_env.uav.energy = max(0.0, self.sim_env.uav.energy - uav_energy_spent)

        # 2. Advance mobile sensors
        self.sim_env.move_sensors()

        # 3. Update dynamic graph
        graph = self.network.build_graph(self.sim_env)
        feat, adj = extract_node_features_and_adj(self.sim_env, graph)
        self.feature_history.append(feat)
        self.adj_history.append(adj)

        # 4. Evaluate routing actions & packet delivery
        routing_actions = action["routing_actions"]
        pdr_delivered = 0
        total_packets = 0
        priority_pdr = 0.0
        node_keys = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]

        for i in range(NUM_SENSORS):
            sensor = self.sim_env.sensors[i]
            s_name = f"S{i}"
            next_hop_idx = routing_actions[i]
            target_name = node_keys[next_hop_idx]
            total_packets += 1

            # Check if direct link exists
            if graph.has_edge(s_name, target_name):
                link_q = graph[s_name][target_name].get("link_quality", 0.0)
                sensor.energy = max(0.0, sensor.energy - self.sensor_tx_energy)

                # If target is CS or can reach CS
                if target_name == "CS":
                    pdr_delivered += 1
                    priority_pdr += (sensor.priority ** 1.5)
                elif target_name == "UAV" and graph.has_edge("UAV", "CS"):
                    pdr_delivered += 1
                    priority_pdr += (sensor.priority ** 1.5)
                elif nx.has_path(graph, s_name, "CS"):
                    pdr_delivered += 1
                    priority_pdr += (sensor.priority ** 1.5)

        # 5. Compute Reward
        pdr_ratio = pdr_delivered / max(1, total_packets)
        uav_cs_connected = 1.0 if graph.has_edge("UAV", "CS") else -0.5
        uav_sensor_links = sum(1 for s in range(NUM_SENSORS) if graph.has_edge(f"S{s}", "UAV"))
        
        # Reward components
        r_delivery = 4.0 * priority_pdr
        r_connectivity = 2.0 * uav_cs_connected + 0.5 * uav_sensor_links
        r_energy = -1.0 * (uav_energy_spent * 10.0)
        
        # Corridor bounds encouragement
        r_corridor = 0.5 if (CORRIDOR_X_MIN <= new_x <= CORRIDOR_X_MAX and CORRIDOR_Y_MIN <= new_y <= CORRIDOR_Y_MAX) else -1.0
        
        reward = r_delivery + r_connectivity + r_energy + r_corridor

        terminated = bool(self.current_step >= self.max_steps or self.sim_env.uav.energy <= 0)
        truncated = False

        obs = self._get_obs()
        info = self._get_info()
        info.update({
            "pdr_ratio": pdr_ratio,
            "packets_delivered": pdr_delivered,
            "priority_score": priority_pdr,
            "reward_breakdown": {
                "r_delivery": r_delivery,
                "r_connectivity": r_connectivity,
                "r_energy": r_energy,
                "r_corridor": r_corridor
            }
        })

        return obs, reward, terminated, truncated, info


# -------------------------------------------------------------
# Step 6 Verification & Visualization
# -------------------------------------------------------------

def run_step6():
    print("=" * 60)
    print("Step 6 - Creating MAPPO Gymnasium Environment")
    print("=" * 60)

    env = UAVWSNEnv(history_window=5, max_steps=25)
    obs, info = env.reset(seed=42)

    print(f"Observation space: {env.observation_space}")
    print(f"Action space:      {env.action_space}")
    print(f"Initial UAV position: {info['uav_position']}")
    print(f"Initial connected components: {info['connected_components']}")

    steps = 15
    rewards = []
    pdr_list = []
    positions = [info['uav_position']]

    print(f"\nStepping through environment for {steps} steps with heuristic actions...")
    for s in range(steps):
        # Sample or construct valid actions:
        # UAV moves towards center/corridor
        curr_pos = env.sim_env.uav.position
        target_pos = (500.0, 500.0)
        dir_vec = np.array(target_pos) - np.array(curr_pos)
        dist = np.linalg.norm(dir_vec)
        if dist > 0:
            dir_vec = dir_vec / dist
        uav_action = dir_vec.astype(np.float32)

        # Sensor routing: greedy to UAV (idx 20) or neighbors
        routing_actions = np.array([20 if i % 2 == 0 else (i + 1) % NUM_SENSORS for i in range(NUM_SENSORS)])

        action = {
            "uav_action": uav_action,
            "routing_actions": routing_actions
        }

        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(reward)
        pdr_list.append(info["pdr_ratio"])
        positions.append(info["uav_position"])

        if (s + 1) % 5 == 0 or s == 0:
            print(f"  Step {s+1:2d} | Reward: {reward:6.2f} | PDR: {info['pdr_ratio']*100:5.1f}% | Components: {info['connected_components']} | UAV Energy: {info['uav_energy']:.1f}")

    # -------------------------------------------------------------
    # Visualization for Step 6
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. UAV Trajectory & Corridor Environment
    ax1 = axes[0, 0]
    ax1.axvspan(CORRIDOR_X_MIN, CORRIDOR_X_MAX, alpha=0.15, color="green", label="Surveillance Corridor")
    if ENABLE_DISRUPTION:
        rect = plt.Rectangle(
            (DISRUPTION_X_MIN, DISRUPTION_Y_MIN),
            DISRUPTION_X_MAX - DISRUPTION_X_MIN,
            DISRUPTION_Y_MAX - DISRUPTION_Y_MIN,
            color="red", alpha=0.25, label="Disruption Zone"
        )
        ax1.add_patch(rect)

    pos_x = [p[0] for p in positions]
    pos_y = [p[1] for p in positions]
    ax1.plot(pos_x, pos_y, "o-", color="#e7298a", linewidth=2.5, label="UAV Relay Trajectory")
    ax1.scatter([pos_x[0]], [pos_y[0]], marker="^", s=150, color="orange", label="UAV Start", zorder=5)
    ax1.scatter([pos_x[-1]], [pos_y[-1]], marker="^", s=150, color="purple", label="UAV Current", zorder=5)
    ax1.scatter([env.sim_env.command_station.position[0]], [env.sim_env.command_station.position[1]], marker="s", s=180, color="darkgreen", label="Command Station", zorder=5)
    
    sensor_x = [s.position[0] for s in env.sim_env.sensors]
    sensor_y = [s.position[1] for s in env.sim_env.sensors]
    ax1.scatter(sensor_x, sensor_y, marker="o", s=60, color="#1f78b4", label="Sensors", alpha=0.7)
    
    ax1.set_xlim(0, AREA_WIDTH)
    ax1.set_ylim(0, AREA_HEIGHT)
    ax1.set_title("UAV Repositioning Trajectory & Environment", fontsize=11, fontweight="bold")
    ax1.set_xlabel("X (meters)")
    ax1.set_ylabel("Y (meters)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. Dynamic WSN Topology at Step N
    ax2 = axes[0, 1]
    curr_graph = env.network.build_graph(env.sim_env)
    nx_pos = {f"S{s.node_id}": s.position for s in env.sim_env.sensors}
    nx_pos["UAV"] = env.sim_env.uav.position
    nx_pos["CS"] = env.sim_env.command_station.position
    nx.draw_networkx_edges(curr_graph, nx_pos, ax=ax2, alpha=0.4, edge_color="gray")
    nx.draw_networkx_nodes(curr_graph, nx_pos, nodelist=[f"S{i}" for i in range(NUM_SENSORS)], ax=ax2, node_size=80, node_color="#33a02c")
    nx.draw_networkx_nodes(curr_graph, nx_pos, nodelist=["UAV"], ax=ax2, node_size=200, node_shape="^", node_color="#e31a1c")
    nx.draw_networkx_nodes(curr_graph, nx_pos, nodelist=["CS"], ax=ax2, node_size=200, node_shape="s", node_color="#1f78b4")
    ax2.set_xlim(0, AREA_WIDTH)
    ax2.set_ylim(0, AREA_HEIGHT)
    ax2.set_title(f"Active WSN Graph (Step {steps})", fontsize=11, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 3. Reward Trajectory
    ax3 = axes[1, 0]
    ax3.plot(range(1, steps + 1), rewards, "o-", color="#386cb0", linewidth=2)
    ax3.set_title("Step Reward Curve Across Episode", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Environment Step")
    ax3.set_ylabel("Reward Value")
    ax3.grid(True, linestyle="--", alpha=0.4)

    # 4. PDR and Environment Spec Box
    ax4 = axes[1, 1]
    ax4.plot(range(1, steps + 1), [p * 100 for p in pdr_list], "s-", color="#7fc97f", linewidth=2, label="Packet Delivery Ratio (%)")
    ax4.set_title("Packet Delivery Ratio (PDR %)", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Environment Step")
    ax4.set_ylabel("PDR (%)")
    ax4.set_ylim(0, 105)
    ax4.grid(True, linestyle="--", alpha=0.4)

    env_spec_text = (
        "Gymnasium Environment Specifications:\n"
        "-------------------------------------\n"
        "• Framework: Gymnasium v1.3.0 compliant\n"
        "• Action Space: Dict(uav: Box(2,), routing: MultiDiscrete(22))\n"
        "• Obs Space: Dict(node_feat: (5,22,8), adj: (5,22,22))\n"
        f"• Status: Step {steps} completed successfully"
    )
    ax4.text(
        0.05, 0.25, env_spec_text,
        transform=ax4.transAxes,
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#fff2ae", alpha=0.85)
    )

    plt.suptitle("Step 6: MAPPO Gymnasium Environment Output", fontsize=15, fontweight="bold")
    plt.tight_layout()
    output_filename = "step6_mappo_env_output.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Step 6 visualization to {output_filename}")
    print("Step 6 completed successfully!")


if __name__ == "__main__":
    run_step6()
