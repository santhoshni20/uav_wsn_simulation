import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import (
    CORRIDOR_X_MIN,
    CORRIDOR_X_MAX,
    CORRIDOR_Y_MIN,
    CORRIDOR_Y_MAX,
    MAX_PRIORITY,
    UAV_COMMUNICATION_RANGE
)


class MultiObjectiveRewardCalculator:
    """
    Step 8: Multi-Objective Mission-Priority Reward Function.
    Balances:
      1. Mission-critical packet delivery (priority-weighted)
      2. End-to-end transmission delay (hop count penalty)
      3. Sensor & UAV energy consumption
      4. Unnecessary UAV movement / oscillation penalty
      5. Connectivity recovery bonus (reconnecting isolated partitions)
      6. Corridor boundary constraints
    """
    def __init__(
        self,
        w_delivery=5.0,
        w_delay=0.4,
        w_energy=1.5,
        w_movement=0.8,
        w_connectivity=3.0,
        w_corridor=2.0
    ):
        self.w_delivery = w_delivery
        self.w_delay = w_delay
        self.w_energy = w_energy
        self.w_movement = w_movement
        self.w_connectivity = w_connectivity
        self.w_corridor = w_corridor

    def compute_reward(
        self,
        environment,
        graph,
        routing_actions,
        uav_move_dist,
        uav_energy_spent,
        previous_components=None
    ):
        """
        Computes total reward and individual component breakdown.
        """
        num_sensors = len(environment.sensors)
        node_keys = [f"S{i}" for i in range(num_sensors)] + ["UAV", "CS"]

        # 1. Packet delivery & delay evaluation
        delivered_packets = 0
        priority_delivery_score = 0.0
        total_hops = 0
        mission_critical_delivered = 0

        for i, sensor in enumerate(environment.sensors):
            s_name = f"S{i}"
            target_idx = routing_actions[i]
            target_name = node_keys[target_idx]

            # Route validation
            if graph.has_edge(s_name, target_name):
                # Check path to CS
                if target_name == "CS":
                    delivered_packets += 1
                    hop_count = 1
                    weighted_prio = (sensor.priority / MAX_PRIORITY) ** 2
                    priority_delivery_score += weighted_prio
                    total_hops += hop_count
                    if sensor.priority == MAX_PRIORITY:
                        mission_critical_delivered += 1
                elif target_name == "UAV" and graph.has_edge("UAV", "CS"):
                    delivered_packets += 1
                    hop_count = 2
                    weighted_prio = (sensor.priority / MAX_PRIORITY) ** 2
                    priority_delivery_score += weighted_prio
                    total_hops += hop_count
                    if sensor.priority == MAX_PRIORITY:
                        mission_critical_delivered += 1
                elif nx.has_path(graph, target_name, "CS") or nx.has_path(graph, s_name, "CS"):
                    try:
                        start_node = target_name if nx.has_path(graph, target_name, "CS") else s_name
                        path_len = nx.shortest_path_length(graph, start_node, "CS") + (1 if start_node == target_name else 0)
                        delivered_packets += 1
                        hop_count = path_len
                        weighted_prio = (sensor.priority / MAX_PRIORITY) ** 2
                        priority_delivery_score += weighted_prio
                        total_hops += hop_count
                        if sensor.priority == MAX_PRIORITY:
                            mission_critical_delivered += 1
                    except (nx.NetworkXNoPath, nx.NodeNotFound):
                        pass

        # Component 1: Delivery Reward (quadratic scaling for priority)
        r_delivery = self.w_delivery * priority_delivery_score

        # Component 2: End-to-end Delay Penalty
        avg_hops = (total_hops / delivered_packets) if delivered_packets > 0 else 5.0
        r_delay = -self.w_delay * max(0.0, avg_hops - 1.0)

        # Component 3: Energy Consumption Penalty
        sensor_energy_cost = delivered_packets * 0.02
        total_energy_cost = sensor_energy_cost + uav_energy_spent
        r_energy = -self.w_energy * total_energy_cost

        # Component 4: Unnecessary UAV Movement Penalty
        normalized_move = uav_move_dist / 30.0  # Normalized by max speed
        r_movement = -self.w_movement * (normalized_move ** 2)

        # Component 5: Connectivity Recovery Bonus & Relay Guidance
        current_components = nx.number_connected_components(graph)
        uav_bridges_cs = 2.0 if graph.has_edge("UAV", "CS") else -1.0
        sensors_connected_to_uav = sum(1 for s in range(num_sensors) if graph.has_edge(f"S{s}", "UAV"))
        
        # Guide UAV towards bridge relay position (corridor center between sensors and CS)
        uav_pos = np.array(environment.uav.position)
        target_relay_y = 280.0
        relay_dist = np.abs(uav_pos[1] - target_relay_y) / 500.0
        guidance = -1.0 * relay_dist

        recovery_bonus = 0.0
        if previous_components is not None and current_components < previous_components:
            # Successfully reconnected partitioned network components
            recovery_bonus = float(previous_components - current_components) * 2.0
            
        r_connectivity = self.w_connectivity * (uav_bridges_cs * 0.5 + 0.1 * sensors_connected_to_uav + recovery_bonus + guidance * 0.4)

        # Component 6: Corridor Boundary Constraint Penalty
        ux, uy = environment.uav.position
        in_corridor = (CORRIDOR_X_MIN <= ux <= CORRIDOR_X_MAX and CORRIDOR_Y_MIN <= uy <= CORRIDOR_Y_MAX)
        r_corridor = 0.5 if in_corridor else -self.w_corridor

        # Total scalar reward
        total_reward = r_delivery + r_delay + r_energy + r_movement + r_connectivity + r_corridor

        details = {
            "r_total": total_reward,
            "r_delivery": r_delivery,
            "r_delay": r_delay,
            "r_energy": r_energy,
            "r_movement": r_movement,
            "r_connectivity": r_connectivity,
            "r_corridor": r_corridor,
            "delivered_packets": delivered_packets,
            "mission_critical_delivered": mission_critical_delivered,
            "avg_hops": avg_hops,
            "current_components": current_components
        }
        return total_reward, details


# -------------------------------------------------------------
# Step 8 Verification & Visualization
# -------------------------------------------------------------

def run_step8():
    print("=" * 60)
    print("Step 8 - Defining and Validating Multi-Objective Reward Function")
    print("=" * 60)

    from environment import SimulationEnvironment
    from network import DynamicWSNGraph

    calc = MultiObjectiveRewardCalculator()
    env = SimulationEnvironment()
    network = DynamicWSNGraph()

    # Simulate 4 characteristic scenarios to show sensitivity of each reward component:
    scenarios = [
        ("Ideal Relay & Delivery", (500.0, 350.0), 5.0, 0.06, True, 3),
        ("Erratic Movement & No CS Link", (100.0, 950.0), 30.0, 0.25, False, 5),
        ("Corridor Breach / Out-of-Bounds", (150.0, 50.0), 28.0, 0.22, False, 5),
        ("Partition Recovery", (500.0, 400.0), 10.0, 0.10, True, 2)
    ]

    scenario_names = []
    breakdown_data = {
        "Delivery": [],
        "Delay": [],
        "Energy": [],
        "Movement": [],
        "Connectivity": [],
        "Corridor": [],
        "Total": []
    }

    print("Evaluating Reward Breakdown Across Scenarios:")
    for name, uav_pos, move_dist, energy, connect_cs, comps in scenarios:
        env.uav.position = uav_pos
        graph = network.build_graph(env)
        # Construct routing actions
        actions = np.array([20 if i % 2 == 0 else 21 for i in range(len(env.sensors))])
        
        tot_r, det = calc.compute_reward(
            env, graph, actions, uav_move_dist=move_dist, uav_energy_spent=energy, previous_components=comps + 1
        )
        scenario_names.append(name)
        breakdown_data["Delivery"].append(det["r_delivery"])
        breakdown_data["Delay"].append(det["r_delay"])
        breakdown_data["Energy"].append(det["r_energy"])
        breakdown_data["Movement"].append(det["r_movement"])
        breakdown_data["Connectivity"].append(det["r_connectivity"])
        breakdown_data["Corridor"].append(det["r_corridor"])
        breakdown_data["Total"].append(tot_r)

        print(f"  [{name}]")
        print(f"    Total: {tot_r:6.2f} | Delivery: {det['r_delivery']:5.2f} | Connect: {det['r_connectivity']:5.2f} | Delay: {det['r_delay']:5.2f} | Energy: {det['r_energy']:5.2f}")

    # -------------------------------------------------------------
    # Visualization for Step 8
    # -------------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # 1. Stacked / Grouped Bar Chart of Reward Components
    ax1 = axes[0, 0]
    x = np.arange(len(scenario_names))
    width = 0.12
    colors = ["#2ca02c", "#d62728", "#ff7f0e", "#9467bd", "#1f77b4", "#8c564b"]
    keys = ["Delivery", "Delay", "Energy", "Movement", "Connectivity", "Corridor"]
    
    for idx, key in enumerate(keys):
        ax1.bar(x + (idx - 2.5) * width, breakdown_data[key], width, label=key, color=colors[idx], alpha=0.85)

    ax1.set_xticks(x)
    ax1.set_xticklabels(scenario_names, rotation=15, fontsize=8.5)
    ax1.set_title("Multi-Objective Reward Components Breakdown", fontsize=11, fontweight="bold")
    ax1.set_ylabel("Reward Value")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. Total Net Reward across Scenarios
    ax2 = axes[0, 1]
    bar_colors = ["#2b83ba" if v > 0 else "#d7191c" for v in breakdown_data["Total"]]
    bars = ax2.bar(scenario_names, breakdown_data["Total"], color=bar_colors, alpha=0.85, width=0.5)
    for bar, v in zip(bars, breakdown_data["Total"]):
        ax2.text(bar.get_x() + bar.get_width()/2, v + (0.3 if v > 0 else -0.8), f"{v:.1f}", ha="center", fontweight="bold", fontsize=9)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.6)
    ax2.set_xticks(range(len(scenario_names)))
    ax2.set_xticklabels(scenario_names, rotation=15, fontsize=8.5)
    ax2.set_title("Net Total Scalar Reward $R_{total}$", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Total Reward")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 3. Priority Scaling Function Curve
    ax3 = axes[1, 0]
    priorities = [1, 2, 3]
    linear_scale = [p / 3.0 for p in priorities]
    quad_scale = [(p / 3.0) ** 2 for p in priorities]
    cube_scale = [(p / 3.0) ** 3 for p in priorities]
    ax3.plot(priorities, linear_scale, "o--", color="gray", label="Linear $(P/3)$")
    ax3.plot(priorities, quad_scale, "s-", color="#e41a1c", linewidth=2.5, label="Quadratic $(P/3)^2$ (Proposed)")
    ax3.plot(priorities, cube_scale, "^:", color="#377eb8", label="Cubic $(P/3)^3$")
    ax3.set_xticks(priorities)
    ax3.set_xticklabels(["Low (P1)", "Medium (P2)", "Critical (P3)"], fontsize=9)
    ax3.set_title("Mission-Critical Priority Weighting Scaling", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Packet Priority Level")
    ax3.set_ylabel("Delivery Reward Multiplier")
    ax3.legend(loc="upper left")
    ax3.grid(True, linestyle="--", alpha=0.3)

    # 4. Mathematical Formulation & Properties Box
    ax4 = axes[1, 1]
    ax4.axis("off")
    formula_text = (
        "Multi-Objective Reward Formulation (Step 8):\n"
        "============================================\n"
        "R_total = R_delivery + R_delay + R_energy\n"
        "        + R_movement + R_connectivity + R_corridor\n\n"
        "• R_delivery = w_p * Σ (priority_i / 3.0)^2\n"
        "  -> Strongly prioritizes critical mission packets.\n"
        "• R_delay = -w_d * max(0, avg_hops - 1)\n"
        "  -> Minimizes unnecessary multi-hop latency.\n"
        "• R_energy = -w_e * (E_sensors + E_uav)\n"
        "  -> Prevents battery depletion.\n"
        "• R_movement = -w_m * (||Δp_uav|| / v_max)^2\n"
        "  -> Suppresses erratic UAV trajectory.\n"
        "• R_connectivity = +w_c * (Bridges_CS + Reconnected)\n"
        "  -> Rewards restoring partitioned subgraphs.\n"
        "• R_corridor = +0.5 (in corridor) or -w_b (out)\n"
        "  -> Confines UAV to mountain pass bounds."
    )
    ax4.text(
        0.05, 0.5, formula_text,
        verticalalignment="center",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.8", facecolor="#f7f7f7", edgecolor="#cccccc", lw=1.5)
    )

    plt.suptitle("Step 8: Multi-Objective Reward Function Specification", fontsize=15, fontweight="bold")
    plt.tight_layout()
    output_filename = "step8_reward_function_output.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Step 8 visualization to {output_filename}")
    print("Step 8 completed successfully!")


if __name__ == "__main__":
    run_step8()
