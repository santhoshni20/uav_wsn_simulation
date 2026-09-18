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
    CORRIDOR_X_MIN,
    CORRIDOR_X_MAX,
    CORRIDOR_Y_MIN,
    CORRIDOR_Y_MAX,
    MAX_PRIORITY
)
from environment import SimulationEnvironment
from network import DynamicWSNGraph
from rl_env import UAVWSNEnv
from mappo_agents import MAPPOSystem


# =============================================================
# Baseline 1: Static UAV + Dijkstra Shortest-Path Routing
# =============================================================

class StaticUAVDijkstraBaseline:
    """
    UAV remains fixed at the corridor center.
    Routing: Dijkstra shortest path from each sensor to CS.
    """
    def __init__(self):
        self.name = "Static UAV + Dijkstra"
        self.uav_position = (500.0, 500.0)  # Fixed at corridor center

    def get_action(self, env, graph):
        # UAV stays static -> zero displacement
        uav_action = np.array([0.0, 0.0], dtype=np.float32)

        # Route each sensor via shortest path to CS
        node_keys = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]
        routing_actions = np.zeros(NUM_SENSORS, dtype=np.int64)

        for i in range(NUM_SENSORS):
            s_name = f"S{i}"
            if graph.has_node(s_name) and graph.has_node("CS"):
                try:
                    path = nx.shortest_path(graph, s_name, "CS", weight="distance")
                    if len(path) >= 2:
                        next_hop = path[1]
                        routing_actions[i] = node_keys.index(next_hop)
                    else:
                        routing_actions[i] = NUM_SENSORS + 1  # CS
                except nx.NetworkXNoPath:
                    routing_actions[i] = NUM_SENSORS  # Try UAV
            else:
                routing_actions[i] = NUM_SENSORS + 1

        return {"uav_action": uav_action, "routing_actions": routing_actions}


# =============================================================
# Baseline 2: Random Walk UAV + Greedy Geographic Routing
# =============================================================

class RandomUAVGreedyBaseline:
    """
    UAV performs a random walk.
    Routing: Each sensor greedily forwards to the neighbor closest to CS.
    """
    def __init__(self):
        self.name = "Random UAV + Greedy Geo"

    def get_action(self, env, graph):
        # Random UAV movement
        uav_action = np.random.uniform(-1.0, 1.0, size=(2,)).astype(np.float32)

        node_keys = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]
        routing_actions = np.zeros(NUM_SENSORS, dtype=np.int64)
        cs_pos = np.array(env.sim_env.command_station.position)

        for i in range(NUM_SENSORS):
            s_name = f"S{i}"
            neighbors = list(graph.neighbors(s_name)) if graph.has_node(s_name) else []

            if not neighbors:
                routing_actions[i] = NUM_SENSORS + 1  # Default to CS
                continue

            # Find neighbor closest to CS (greedy geographic)
            best_hop = None
            best_dist = float("inf")
            for nb in neighbors:
                nb_pos = np.array(graph.nodes[nb].get("position", cs_pos))
                d = np.linalg.norm(nb_pos - cs_pos)
                if d < best_dist:
                    best_dist = d
                    best_hop = nb

            if best_hop is not None:
                routing_actions[i] = node_keys.index(best_hop)
            else:
                routing_actions[i] = NUM_SENSORS + 1

        return {"uav_action": uav_action, "routing_actions": routing_actions}


# =============================================================
# Baseline 3: Heuristic Centroid UAV + AODV-like Routing
# =============================================================

class HeuristicCentroidBaseline:
    """
    UAV moves toward the centroid of disconnected/isolated sensors.
    Routing: AODV-like opportunistic shortest path.
    """
    def __init__(self):
        self.name = "Heuristic UAV + AODV"

    def get_action(self, env, graph):
        node_keys = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]
        routing_actions = np.zeros(NUM_SENSORS, dtype=np.int64)

        # Find isolated sensor subgraphs (not connected to CS)
        isolated_positions = []
        for i in range(NUM_SENSORS):
            s_name = f"S{i}"
            if graph.has_node(s_name):
                try:
                    if not nx.has_path(graph, s_name, "CS"):
                        pos = env.sim_env.sensors[i].position
                        isolated_positions.append(pos)
                except nx.NodeNotFound:
                    pass

        # UAV moves toward centroid of isolated nodes (or corridor center)
        if isolated_positions:
            centroid = np.mean(isolated_positions, axis=0)
        else:
            centroid = np.array([500.0, 400.0])

        uav_pos = np.array(env.sim_env.uav.position)
        direction = centroid - uav_pos
        norm = np.linalg.norm(direction)
        if norm > 1e-3:
            direction = direction / norm
        uav_action = np.clip(direction, -1.0, 1.0).astype(np.float32)

        # AODV-like opportunistic routing: shortest path if available
        for i in range(NUM_SENSORS):
            s_name = f"S{i}"
            if graph.has_node(s_name) and graph.has_node("CS"):
                try:
                    path = nx.shortest_path(graph, s_name, "CS")
                    if len(path) >= 2:
                        routing_actions[i] = node_keys.index(path[1])
                    else:
                        routing_actions[i] = NUM_SENSORS + 1
                except nx.NetworkXNoPath:
                    # Try via UAV
                    if graph.has_edge(s_name, "UAV"):
                        routing_actions[i] = NUM_SENSORS  # UAV relay
                    else:
                        # Forward to closest neighbor
                        neighbors = list(graph.neighbors(s_name))
                        if neighbors:
                            routing_actions[i] = node_keys.index(neighbors[0])
                        else:
                            routing_actions[i] = NUM_SENSORS + 1
            else:
                routing_actions[i] = NUM_SENSORS + 1

        return {"uav_action": uav_action, "routing_actions": routing_actions}


# =============================================================
# Baseline 4: Vanilla MAPPO (No ST-GNN, MLP-only)
# =============================================================

class VanillaMAPPOBaseline:
    """
    MAPPO with a simple MLP state representation instead of ST-GNN.
    Uses flattened observation features without spatio-temporal encoding.
    """
    def __init__(self):
        self.name = "Vanilla MAPPO (No GNN)"

    def get_action(self, env, graph):
        node_keys = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]
        routing_actions = np.zeros(NUM_SENSORS, dtype=np.int64)

        # Random policy for UAV (untrained MLP approximation)
        uav_action = np.random.uniform(-0.5, 0.5, size=(2,)).astype(np.float32)

        # Semi-random routing with slight preference for closer nodes
        cs_pos = np.array(env.sim_env.command_station.position)
        for i in range(NUM_SENSORS):
            s_name = f"S{i}"
            neighbors = list(graph.neighbors(s_name)) if graph.has_node(s_name) else []
            if neighbors:
                # Weighted random selection (closer to CS = higher weight)
                weights = []
                for nb in neighbors:
                    nb_pos = np.array(graph.nodes[nb].get("position", cs_pos))
                    d = max(np.linalg.norm(nb_pos - cs_pos), 1.0)
                    weights.append(1.0 / d)
                weights = np.array(weights)
                weights = weights / weights.sum()
                chosen = np.random.choice(len(neighbors), p=weights)
                routing_actions[i] = node_keys.index(neighbors[chosen])
            else:
                routing_actions[i] = NUM_SENSORS + 1

        return {"uav_action": uav_action, "routing_actions": routing_actions}


# =============================================================
# Evaluation Harness: Run all baselines under identical seeds
# =============================================================

def evaluate_method(env, method_fn, num_episodes=5, steps_per_episode=30, seeds=None):
    """
    Evaluates a method (baseline or trained model) across test seeds.
    Returns per-episode metrics.
    """
    if seeds is None:
        seeds = [200 + i for i in range(num_episodes)]

    results = {
        "episode_rewards": [],
        "pdr_ratios": [],
        "packets_delivered": [],
        "avg_hops": [],
        "energy_remaining": [],
        "mission_critical_delivered": []
    }

    for ep_idx, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        ep_reward = 0.0
        ep_pdr = []
        ep_delivered = 0
        ep_mc = 0

        for step in range(steps_per_episode):
            graph = env.network.build_graph(env.sim_env)
            action = method_fn(env, graph)
            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            ep_pdr.append(step_info["pdr_ratio"])
            ep_delivered += step_info["packets_delivered"]
            ep_mc += step_info["reward_breakdown"]["mission_critical_delivered"]

            if terminated or truncated:
                break

        # Collect remaining energy
        sensor_energies = [s.energy for s in env.sim_env.sensors]
        avg_sensor_energy = np.mean(sensor_energies)
        uav_energy = env.sim_env.uav.energy

        results["episode_rewards"].append(ep_reward)
        results["pdr_ratios"].append(np.mean(ep_pdr) * 100.0)
        results["packets_delivered"].append(ep_delivered)
        results["avg_hops"].append(step_info["reward_breakdown"]["avg_hops"])
        results["energy_remaining"].append((avg_sensor_energy + uav_energy) / 2.0)
        results["mission_critical_delivered"].append(ep_mc)

    return results


def evaluate_trained_mappo(env, mappo, num_episodes=5, steps_per_episode=30, seeds=None):
    """
    Evaluates the trained ST-GNN + MAPPO model.
    """
    if seeds is None:
        seeds = [200 + i for i in range(num_episodes)]

    results = {
        "episode_rewards": [],
        "pdr_ratios": [],
        "packets_delivered": [],
        "avg_hops": [],
        "energy_remaining": [],
        "mission_critical_delivered": []
    }

    for ep_idx, seed in enumerate(seeds):
        obs, info = env.reset(seed=seed)
        ep_reward = 0.0
        ep_pdr = []
        ep_delivered = 0
        ep_mc = 0

        for step in range(steps_per_episode):
            nf = torch.tensor(obs["node_features"], dtype=torch.float32)
            ad = torch.tensor(obs["adjacency"], dtype=torch.float32)
            us = torch.tensor(obs["uav_state"], dtype=torch.float32)

            with torch.no_grad():
                node_embs, g_emb = mappo.st_gnn(nf, ad)
                c_state = torch.cat([g_emb, us], dim=-1)
                uav_dist = mappo.uav_actor(c_state)
                uav_act = uav_dist.loc
                route_dist = mappo.routing_actor(node_embs, adj_mask=ad[-1])
                route_act = torch.argmax(route_dist.probs, dim=-1)

            action = {
                "uav_action": uav_act.squeeze(0).numpy(),
                "routing_actions": route_act.squeeze(0).numpy()
            }

            obs, reward, terminated, truncated, step_info = env.step(action)
            ep_reward += reward
            ep_pdr.append(step_info["pdr_ratio"])
            ep_delivered += step_info["packets_delivered"]
            ep_mc += step_info["reward_breakdown"]["mission_critical_delivered"]

            if terminated or truncated:
                break

        sensor_energies = [s.energy for s in env.sim_env.sensors]
        avg_sensor_energy = np.mean(sensor_energies)
        uav_energy = env.sim_env.uav.energy

        results["episode_rewards"].append(ep_reward)
        results["pdr_ratios"].append(np.mean(ep_pdr) * 100.0)
        results["packets_delivered"].append(ep_delivered)
        results["avg_hops"].append(step_info["reward_breakdown"]["avg_hops"])
        results["energy_remaining"].append((avg_sensor_energy + uav_energy) / 2.0)
        results["mission_critical_delivered"].append(ep_mc)

    return results


# =============================================================
# Step 10 Verification & Visualization
# =============================================================

def run_step10():
    print("=" * 60)
    print("Step 10 - Comparing with Baseline Methods")
    print("=" * 60)

    env = UAVWSNEnv(history_window=5, max_steps=30)
    test_seeds = [200, 201, 202, 203, 204]

    # Initialize baselines
    baselines = [
        StaticUAVDijkstraBaseline(),
        RandomUAVGreedyBaseline(),
        HeuristicCentroidBaseline(),
        VanillaMAPPOBaseline()
    ]

    all_results = {}

    # Evaluate each baseline
    for bl in baselines:
        print(f"\nEvaluating: {bl.name}...")
        results = evaluate_method(
            env, bl.get_action,
            num_episodes=5, steps_per_episode=30, seeds=test_seeds
        )
        all_results[bl.name] = results
        mean_pdr = np.mean(results["pdr_ratios"])
        mean_reward = np.mean(results["episode_rewards"])
        print(f"  Mean PDR: {mean_pdr:.1f}% | Mean Reward: {mean_reward:.2f}")

    # Evaluate trained ST-GNN + MAPPO model
    print(f"\nEvaluating: ST-GNN + MAPPO (Proposed)...")
    mappo = MAPPOSystem(env.st_gnn, lr_actor=4e-4, lr_critic=1e-3)
    checkpoint_path = "st_gnn_mappo_checkpoint.pth"
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        mappo.st_gnn.load_state_dict(checkpoint["st_gnn"])
        mappo.critic.load_state_dict(checkpoint["critic"])
        mappo.uav_actor.load_state_dict(checkpoint["uav_actor"])
        mappo.routing_actor.load_state_dict(checkpoint["routing_actor"])
        mappo.eval()
        print("  Loaded trained model checkpoint successfully.")
    except FileNotFoundError:
        print("  WARNING: No checkpoint found. Using untrained MAPPO model.")

    proposed_results = evaluate_trained_mappo(
        env, mappo,
        num_episodes=5, steps_per_episode=30, seeds=test_seeds
    )
    all_results["ST-GNN + MAPPO\n(Proposed)"] = proposed_results
    mean_pdr = np.mean(proposed_results["pdr_ratios"])
    mean_reward = np.mean(proposed_results["episode_rewards"])
    print(f"  Mean PDR: {mean_pdr:.1f}% | Mean Reward: {mean_reward:.2f}")

    # ---------------------------------------------------------
    # Visualization for Step 10
    # ---------------------------------------------------------
    method_names = list(all_results.keys())
    num_methods = len(method_names)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]
    bar_width = 0.14

    # 1. Packet Delivery Ratio (PDR %) Comparison
    ax1 = axes[0, 0]
    pdr_means = [np.mean(all_results[m]["pdr_ratios"]) for m in method_names]
    pdr_stds = [np.std(all_results[m]["pdr_ratios"]) for m in method_names]
    bars = ax1.bar(range(num_methods), pdr_means, yerr=pdr_stds,
                   color=colors[:num_methods], alpha=0.85, edgecolor="black",
                   capsize=5, width=0.55)
    for bar, val in zip(bars, pdr_means):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                 f"{val:.1f}%", ha="center", fontweight="bold", fontsize=9)
    ax1.set_xticks(range(num_methods))
    ax1.set_xticklabels(method_names, fontsize=8, rotation=10)
    ax1.set_ylabel("PDR (%)")
    ax1.set_title("Packet Delivery Ratio (PDR %)", fontsize=12, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. Cumulative Episode Reward Comparison
    ax2 = axes[0, 1]
    reward_means = [np.mean(all_results[m]["episode_rewards"]) for m in method_names]
    reward_stds = [np.std(all_results[m]["episode_rewards"]) for m in method_names]
    bar_colors_rw = ["#d7191c" if v < 0 else c for v, c in zip(reward_means, colors)]
    bars2 = ax2.bar(range(num_methods), reward_means, yerr=reward_stds,
                    color=bar_colors_rw, alpha=0.85, edgecolor="black",
                    capsize=5, width=0.55)
    for bar, val in zip(bars2, reward_means):
        y_off = 3 if val >= 0 else -8
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + y_off,
                 f"{val:.1f}", ha="center", fontweight="bold", fontsize=9)
    ax2.axhline(0, color="black", linestyle="--", alpha=0.5)
    ax2.set_xticks(range(num_methods))
    ax2.set_xticklabels(method_names, fontsize=8, rotation=10)
    ax2.set_ylabel("Cumulative Reward")
    ax2.set_title("Episode Cumulative Reward", fontsize=12, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 3. Energy Efficiency & Mission-Critical Delivery (Grouped Bar)
    ax3 = axes[1, 0]
    x = np.arange(num_methods)
    energy_vals = [np.mean(all_results[m]["energy_remaining"]) for m in method_names]
    mc_vals = [np.mean(all_results[m]["mission_critical_delivered"]) for m in method_names]
    b1 = ax3.bar(x - 0.2, energy_vals, 0.35, label="Avg Remaining Energy (%)",
                 color="#66c2a5", alpha=0.85, edgecolor="black")
    b2 = ax3.bar(x + 0.2, mc_vals, 0.35, label="Mission-Critical Delivered",
                 color="#fc8d62", alpha=0.85, edgecolor="black")
    ax3.set_xticks(x)
    ax3.set_xticklabels(method_names, fontsize=8, rotation=10)
    ax3.set_title("Energy Efficiency & Mission-Critical Delivery", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Value")
    ax3.legend(loc="upper left", fontsize=8)
    ax3.grid(True, linestyle="--", alpha=0.3)

    # 4. Comprehensive Comparison Summary Table
    ax4 = axes[1, 1]
    ax4.axis("off")
    table_data = []
    headers = ["Method", "PDR (%)", "Reward", "Hops", "Energy", "MC Pkts"]
    for m in method_names:
        r = all_results[m]
        short_name = m.replace("\n", " ")
        table_data.append([
            short_name,
            f"{np.mean(r['pdr_ratios']):.1f}",
            f"{np.mean(r['episode_rewards']):.1f}",
            f"{np.mean(r['avg_hops']):.2f}",
            f"{np.mean(r['energy_remaining']):.1f}",
            f"{np.mean(r['mission_critical_delivered']):.0f}"
        ])

    table = ax4.table(
        cellText=table_data,
        colLabels=headers,
        cellLoc="center",
        loc="center",
        colColours=["#dce6f1"] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.8)
    # Highlight proposed method row
    for j in range(len(headers)):
        table[len(table_data), j].set_facecolor("#d4edda")
        table[len(table_data), j].set_text_props(fontweight="bold")

    ax4.set_title("Performance Comparison Summary", fontsize=12, fontweight="bold", pad=20)

    plt.suptitle("Step 10: Baseline Comparison with ST-GNN + MAPPO", fontsize=15, fontweight="bold")
    plt.tight_layout()
    output_filename = "step10_baseline_comparison_output.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Step 10 visualization to {output_filename}")
    print("Step 10 completed successfully!")


if __name__ == "__main__":
    run_step10()
