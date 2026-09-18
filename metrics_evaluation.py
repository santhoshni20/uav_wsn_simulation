import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import torch

from config import (
    NUM_SENSORS,
    MAX_PRIORITY,
    AREA_WIDTH,
    AREA_HEIGHT,
    CORRIDOR_X_MIN,
    CORRIDOR_X_MAX,
    CORRIDOR_Y_MIN,
    CORRIDOR_Y_MAX
)
from rl_env import UAVWSNEnv
from mappo_agents import MAPPOSystem
from baselines import (
    StaticUAVDijkstraBaseline,
    RandomUAVGreedyBaseline,
    HeuristicCentroidBaseline,
    VanillaMAPPOBaseline
)


# =============================================================
# Comprehensive 8-Metric Evaluation Engine
# =============================================================

class PerformanceEvaluator:
    """
    Evaluates UAV-WSN methods across 8 quantitative metrics:
      1. Packet Delivery Ratio (PDR %)
      2. End-to-End Delay (avg hop count)
      3. Sensor Energy Consumption (%)
      4. UAV Energy Consumption (%)
      5. Network Lifetime (steps until first sensor dies)
      6. Throughput (packets delivered per timestep)
      7. Packet Loss Rate (%)
      8. Mission-Critical Packet Delivery Rate (%)
    """

    def __init__(self, env, steps_per_episode=30, num_episodes=5):
        self.env = env
        self.steps_per_episode = steps_per_episode
        self.num_episodes = num_episodes
        self.test_seeds = [300 + i for i in range(num_episodes)]

    def evaluate_baseline(self, method_fn, name="Baseline"):
        """Evaluate a baseline method returning dict of 8 metrics."""
        metrics = {
            "pdr": [], "delay": [], "sensor_energy": [], "uav_energy": [],
            "lifetime": [], "throughput": [], "packet_loss": [], "mc_delivery": []
        }

        for seed in self.test_seeds:
            obs, info = self.env.reset(seed=seed)
            ep_delivered = 0
            ep_total = 0
            ep_hops = []
            ep_mc_total = 0
            ep_mc_delivered = 0
            initial_sensor_energies = [s.energy for s in self.env.sim_env.sensors]
            initial_uav_energy = self.env.sim_env.uav.energy
            lifetime_step = self.steps_per_episode  # Default: survived entire episode

            for step in range(self.steps_per_episode):
                graph = self.env.network.build_graph(self.env.sim_env)
                action = method_fn(self.env, graph)
                obs, reward, terminated, truncated, step_info = self.env.step(action)

                delivered = step_info["packets_delivered"]
                ep_delivered += delivered
                ep_total += NUM_SENSORS

                # Track hops
                avg_hops = step_info["reward_breakdown"]["avg_hops"]
                if delivered > 0:
                    ep_hops.append(avg_hops)

                # Mission-critical tracking
                for s in self.env.sim_env.sensors:
                    if s.priority == MAX_PRIORITY:
                        ep_mc_total += 1
                ep_mc_delivered += step_info["reward_breakdown"]["mission_critical_delivered"]

                # Check network lifetime (first sensor to die)
                for s in self.env.sim_env.sensors:
                    if s.energy <= 0 and lifetime_step == self.steps_per_episode:
                        lifetime_step = step + 1

                if terminated or truncated:
                    break

            # Compute metrics for this episode
            pdr = (ep_delivered / max(1, ep_total)) * 100.0
            avg_delay = np.mean(ep_hops) if ep_hops else 5.0
            sensor_consumed = np.mean([(ie - s.energy) for ie, s in
                                        zip(initial_sensor_energies, self.env.sim_env.sensors)])
            uav_consumed = initial_uav_energy - self.env.sim_env.uav.energy
            throughput = ep_delivered / self.steps_per_episode
            loss_rate = ((ep_total - ep_delivered) / max(1, ep_total)) * 100.0
            mc_rate = (ep_mc_delivered / max(1, ep_mc_total)) * 100.0

            metrics["pdr"].append(pdr)
            metrics["delay"].append(avg_delay)
            metrics["sensor_energy"].append(sensor_consumed)
            metrics["uav_energy"].append(uav_consumed)
            metrics["lifetime"].append(lifetime_step)
            metrics["throughput"].append(throughput)
            metrics["packet_loss"].append(loss_rate)
            metrics["mc_delivery"].append(mc_rate)

        return {k: np.array(v) for k, v in metrics.items()}

    def evaluate_mappo(self, mappo):
        """Evaluate trained MAPPO model across same test seeds."""
        metrics = {
            "pdr": [], "delay": [], "sensor_energy": [], "uav_energy": [],
            "lifetime": [], "throughput": [], "packet_loss": [], "mc_delivery": []
        }

        for seed in self.test_seeds:
            obs, info = self.env.reset(seed=seed)
            ep_delivered = 0
            ep_total = 0
            ep_hops = []
            ep_mc_total = 0
            ep_mc_delivered = 0
            initial_sensor_energies = [s.energy for s in self.env.sim_env.sensors]
            initial_uav_energy = self.env.sim_env.uav.energy
            lifetime_step = self.steps_per_episode

            for step in range(self.steps_per_episode):
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

                obs, reward, terminated, truncated, step_info = self.env.step(action)

                delivered = step_info["packets_delivered"]
                ep_delivered += delivered
                ep_total += NUM_SENSORS

                avg_hops = step_info["reward_breakdown"]["avg_hops"]
                if delivered > 0:
                    ep_hops.append(avg_hops)

                for s in self.env.sim_env.sensors:
                    if s.priority == MAX_PRIORITY:
                        ep_mc_total += 1
                ep_mc_delivered += step_info["reward_breakdown"]["mission_critical_delivered"]

                for s in self.env.sim_env.sensors:
                    if s.energy <= 0 and lifetime_step == self.steps_per_episode:
                        lifetime_step = step + 1

                if terminated or truncated:
                    break

            pdr = (ep_delivered / max(1, ep_total)) * 100.0
            avg_delay = np.mean(ep_hops) if ep_hops else 5.0
            sensor_consumed = np.mean([(ie - s.energy) for ie, s in
                                        zip(initial_sensor_energies, self.env.sim_env.sensors)])
            uav_consumed = initial_uav_energy - self.env.sim_env.uav.energy
            throughput = ep_delivered / self.steps_per_episode
            loss_rate = ((ep_total - ep_delivered) / max(1, ep_total)) * 100.0
            mc_rate = (ep_mc_delivered / max(1, ep_mc_total)) * 100.0

            metrics["pdr"].append(pdr)
            metrics["delay"].append(avg_delay)
            metrics["sensor_energy"].append(sensor_consumed)
            metrics["uav_energy"].append(uav_consumed)
            metrics["lifetime"].append(lifetime_step)
            metrics["throughput"].append(throughput)
            metrics["packet_loss"].append(loss_rate)
            metrics["mc_delivery"].append(mc_rate)

        return {k: np.array(v) for k, v in metrics.items()}


# =============================================================
# Step 11 Verification & Visualization
# =============================================================

def run_step11():
    print("=" * 60)
    print("Step 11 - Quantitative Performance Evaluation (8 Metrics)")
    print("=" * 60)

    env = UAVWSNEnv(history_window=5, max_steps=30)
    evaluator = PerformanceEvaluator(env, steps_per_episode=30, num_episodes=5)

    # Baselines
    baselines = {
        "Static+Dijkstra": StaticUAVDijkstraBaseline(),
        "Random+Greedy": RandomUAVGreedyBaseline(),
        "Heuristic+AODV": HeuristicCentroidBaseline(),
        "Vanilla MAPPO": VanillaMAPPOBaseline()
    }

    all_metrics = {}

    for name, bl in baselines.items():
        print(f"\nEvaluating: {name}...")
        metrics = evaluator.evaluate_baseline(bl.get_action, name)
        all_metrics[name] = metrics
        print(f"  PDR: {np.mean(metrics['pdr']):.1f}% | Delay: {np.mean(metrics['delay']):.2f} hops | "
              f"Throughput: {np.mean(metrics['throughput']):.2f} pkts/step")

    # Trained ST-GNN + MAPPO
    print(f"\nEvaluating: ST-GNN + MAPPO (Proposed)...")
    mappo = MAPPOSystem(env.st_gnn, lr_actor=4e-4, lr_critic=1e-3)
    try:
        checkpoint = torch.load("st_gnn_mappo_checkpoint.pth", map_location="cpu", weights_only=True)
        mappo.st_gnn.load_state_dict(checkpoint["st_gnn"])
        mappo.critic.load_state_dict(checkpoint["critic"])
        mappo.uav_actor.load_state_dict(checkpoint["uav_actor"])
        mappo.routing_actor.load_state_dict(checkpoint["routing_actor"])
        mappo.eval()
        print("  Loaded trained model checkpoint.")
    except FileNotFoundError:
        print("  WARNING: No checkpoint found.")

    proposed_metrics = evaluator.evaluate_mappo(mappo)
    all_metrics["ST-GNN+MAPPO\n(Proposed)"] = proposed_metrics
    print(f"  PDR: {np.mean(proposed_metrics['pdr']):.1f}% | Delay: {np.mean(proposed_metrics['delay']):.2f} hops | "
          f"Throughput: {np.mean(proposed_metrics['throughput']):.2f} pkts/step")

    method_names = list(all_metrics.keys())
    num_methods = len(method_names)

    # ---------------------------------------------------------
    # Print Full 8-Metric Summary Table
    # ---------------------------------------------------------
    metric_labels = [
        ("pdr", "PDR (%)", "[H]"),
        ("delay", "Avg Delay (hops)", "[L]"),
        ("sensor_energy", "Sensor Energy Used", "[L]"),
        ("uav_energy", "UAV Energy Used", "[L]"),
        ("lifetime", "Network Lifetime (steps)", "[H]"),
        ("throughput", "Throughput (pkts/step)", "[H]"),
        ("packet_loss", "Packet Loss Rate (%)", "[L]"),
        ("mc_delivery", "Mission-Critical PDR (%)", "[H]")
    ]

    print("\n" + "=" * 100)
    print(f"{'Metric':<30s}", end="")
    for m in method_names:
        print(f"  {m.replace(chr(10), ' '):<18s}", end="")
    print()
    print("-" * 100)
    for key, label, direction in metric_labels:
        print(f"{label} {direction:<26s}", end="")
        for m in method_names:
            val = np.mean(all_metrics[m][key])
            print(f"  {val:<18.2f}", end="")
        print()
    print("=" * 100)

    # ---------------------------------------------------------
    # Visualization for Step 11
    # ---------------------------------------------------------
    fig = plt.figure(figsize=(18, 14))
    gs = fig.add_gridspec(3, 3, hspace=0.45, wspace=0.35)

    colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#1f77b4"]
    short_names = [n.replace("\n", " ") for n in method_names]

    # 1. PDR Box Plot
    ax1 = fig.add_subplot(gs[0, 0])
    pdr_data = [all_metrics[m]["pdr"] for m in method_names]
    bp1 = ax1.boxplot(pdr_data, patch_artist=True)
    for patch, color in zip(bp1['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax1.set_title("1. Packet Delivery Ratio (%)", fontsize=10, fontweight="bold")
    ax1.set_ylabel("PDR (%)")
    ax1.set_xticks(range(1, len(short_names) + 1))
    ax1.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. End-to-End Delay
    ax2 = fig.add_subplot(gs[0, 1])
    delay_means = [np.mean(all_metrics[m]["delay"]) for m in method_names]
    delay_stds = [np.std(all_metrics[m]["delay"]) for m in method_names]
    bars2 = ax2.bar(range(num_methods), delay_means, yerr=delay_stds,
                    color=colors, alpha=0.85, edgecolor="black", capsize=4)
    for bar, val in zip(bars2, delay_means):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax2.set_xticks(range(num_methods))
    ax2.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax2.set_title("2. End-to-End Delay (Avg Hops)", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Avg Hop Count")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # 3. Sensor Energy Consumption
    ax3 = fig.add_subplot(gs[0, 2])
    se_means = [np.mean(all_metrics[m]["sensor_energy"]) for m in method_names]
    se_stds = [np.std(all_metrics[m]["sensor_energy"]) for m in method_names]
    bars3 = ax3.bar(range(num_methods), se_means, yerr=se_stds,
                    color=colors, alpha=0.85, edgecolor="black", capsize=4)
    for bar, val in zip(bars3, se_means):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                 f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax3.set_xticks(range(num_methods))
    ax3.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax3.set_title("3. Sensor Energy Consumed", fontsize=10, fontweight="bold")
    ax3.set_ylabel("Energy Units")
    ax3.grid(True, linestyle="--", alpha=0.3)

    # 4. UAV Energy Consumption
    ax4 = fig.add_subplot(gs[1, 0])
    ue_means = [np.mean(all_metrics[m]["uav_energy"]) for m in method_names]
    ue_stds = [np.std(all_metrics[m]["uav_energy"]) for m in method_names]
    bars4 = ax4.bar(range(num_methods), ue_means, yerr=ue_stds,
                    color=colors, alpha=0.85, edgecolor="black", capsize=4)
    for bar, val in zip(bars4, ue_means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                 f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax4.set_xticks(range(num_methods))
    ax4.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax4.set_title("4. UAV Energy Consumed", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Energy Units")
    ax4.grid(True, linestyle="--", alpha=0.3)

    # 5. Network Lifetime
    ax5 = fig.add_subplot(gs[1, 1])
    lt_means = [np.mean(all_metrics[m]["lifetime"]) for m in method_names]
    lt_stds = [np.std(all_metrics[m]["lifetime"]) for m in method_names]
    bars5 = ax5.bar(range(num_methods), lt_means, yerr=lt_stds,
                    color=colors, alpha=0.85, edgecolor="black", capsize=4)
    for bar, val in zip(bars5, lt_means):
        ax5.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                 f"{val:.0f}", ha="center", fontsize=8, fontweight="bold")
    ax5.set_xticks(range(num_methods))
    ax5.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax5.set_title("5. Network Lifetime (Steps)", fontsize=10, fontweight="bold")
    ax5.set_ylabel("Timesteps")
    ax5.grid(True, linestyle="--", alpha=0.3)

    # 6. Throughput
    ax6 = fig.add_subplot(gs[1, 2])
    tp_means = [np.mean(all_metrics[m]["throughput"]) for m in method_names]
    tp_stds = [np.std(all_metrics[m]["throughput"]) for m in method_names]
    bars6 = ax6.bar(range(num_methods), tp_means, yerr=tp_stds,
                    color=colors, alpha=0.85, edgecolor="black", capsize=4)
    for bar, val in zip(bars6, tp_means):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                 f"{val:.2f}", ha="center", fontsize=8, fontweight="bold")
    ax6.set_xticks(range(num_methods))
    ax6.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax6.set_title("6. Throughput (Pkts/Step)", fontsize=10, fontweight="bold")
    ax6.set_ylabel("Packets/Step")
    ax6.grid(True, linestyle="--", alpha=0.3)

    # 7. Packet Loss Rate
    ax7 = fig.add_subplot(gs[2, 0])
    pl_means = [np.mean(all_metrics[m]["packet_loss"]) for m in method_names]
    pl_stds = [np.std(all_metrics[m]["packet_loss"]) for m in method_names]
    bars7 = ax7.bar(range(num_methods), pl_means, yerr=pl_stds,
                    color=colors, alpha=0.85, edgecolor="black", capsize=4)
    for bar, val in zip(bars7, pl_means):
        ax7.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax7.set_xticks(range(num_methods))
    ax7.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax7.set_title("7. Packet Loss Rate (%)", fontsize=10, fontweight="bold")
    ax7.set_ylabel("Loss Rate (%)")
    ax7.grid(True, linestyle="--", alpha=0.3)

    # 8. Mission-Critical Delivery Rate
    ax8 = fig.add_subplot(gs[2, 1])
    mc_means = [np.mean(all_metrics[m]["mc_delivery"]) for m in method_names]
    mc_stds = [np.std(all_metrics[m]["mc_delivery"]) for m in method_names]
    bars8 = ax8.bar(range(num_methods), mc_means, yerr=mc_stds,
                    color=colors, alpha=0.85, edgecolor="black", capsize=4)
    for bar, val in zip(bars8, mc_means):
        ax8.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                 f"{val:.1f}%", ha="center", fontsize=8, fontweight="bold")
    ax8.set_xticks(range(num_methods))
    ax8.set_xticklabels(short_names, fontsize=7, rotation=15)
    ax8.set_title("8. Mission-Critical Delivery (%)", fontsize=10, fontweight="bold")
    ax8.set_ylabel("MC PDR (%)")
    ax8.grid(True, linestyle="--", alpha=0.3)

    # 9. Summary Table (lower right)
    ax9 = fig.add_subplot(gs[2, 2])
    ax9.axis("off")
    summary_text = (
        "Performance Evaluation Summary\n"
        "==============================\n"
        "8 Metrics Across 5 Methods:\n\n"
        "Higher is Better:\n"
        "  * PDR, Lifetime, Throughput,\n"
        "    Mission-Critical Delivery\n\n"
        "Lower is Better:\n"
        "  * Delay, Sensor Energy,\n"
        "    UAV Energy, Packet Loss\n\n"
        f"Test Episodes: {evaluator.num_episodes}\n"
        f"Steps/Episode: {evaluator.steps_per_episode}\n"
        f"Sensors: {NUM_SENSORS} mobile nodes"
    )
    ax9.text(0.05, 0.5, summary_text, verticalalignment="center",
             fontsize=9.5, family="monospace",
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f0f0f0", edgecolor="#999", lw=1.5))

    plt.suptitle("Step 11: Quantitative Performance Evaluation (8 Metrics)", fontsize=16, fontweight="bold")
    output_filename = "step11_performance_evaluation.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Step 11 visualization to {output_filename}")
    print("Step 11 completed successfully!")


if __name__ == "__main__":
    run_step11()
