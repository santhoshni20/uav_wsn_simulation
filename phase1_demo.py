"""
============================================================
PHASE 1: Environment & Data Pipeline - Complete Demonstration
============================================================
Step 1 - Simulation Environment
Step 2 - Mobile Sensor Simulation
Step 3 - Dynamic WSN Graph
Step 4 - Communication Disruptions

Run:  python phase1_demo.py
============================================================
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

from config import (
    AREA_WIDTH, AREA_HEIGHT,
    NUM_SENSORS,
    SENSOR_COMMUNICATION_RANGE,
    UAV_COMMUNICATION_RANGE,
    CORRIDOR_X_MIN, CORRIDOR_X_MAX,
    CORRIDOR_Y_MIN, CORRIDOR_Y_MAX,
    ENABLE_DISRUPTION,
    DISRUPTION_X_MIN, DISRUPTION_X_MAX,
    DISRUPTION_Y_MIN, DISRUPTION_Y_MAX,
    MAX_PRIORITY, TOTAL_STEPS
)
from environment import SimulationEnvironment
from network import DynamicWSNGraph


# ============================================================
# Configuration
# ============================================================

np.random.seed(42)

SIMULATION_STEPS = 50          # Total timesteps to simulate
SNAPSHOT_TIMESTEPS = [0, 10, 25, 49]  # Timesteps to capture graph snapshots


# ============================================================
# STEP 1 - Create the Simulation Environment
# ============================================================

def run_step1():
    """Create and display the mountain-pass surveillance environment."""

    print("=" * 65)
    print("  STEP 1 - Simulation Environment")
    print("=" * 65)

    env = SimulationEnvironment()

    print(f"  Area Dimensions      : {AREA_WIDTH} x {AREA_HEIGHT} meters")
    print(f"  Surveillance Corridor: X[{CORRIDOR_X_MIN}-{CORRIDOR_X_MAX}], Y[{CORRIDOR_Y_MIN}-{CORRIDOR_Y_MAX}]")
    print(f"  Number of Sensors    : {NUM_SENSORS}")
    print(f"  UAV Initial Position : {env.uav.position}")
    print(f"  UAV Energy           : {env.uav.energy}")
    print(f"  Command Station      : {env.command_station.position}")
    print(f"  Disruption Zone      : X[{DISRUPTION_X_MIN}-{DISRUPTION_X_MAX}], Y[{DISRUPTION_Y_MIN}-{DISRUPTION_Y_MAX}]")
    print()

    print("  Initial Sensor Deployment:")
    print(f"  {'ID':<4} {'Position':<22} {'Velocity':<10} {'Energy':<10} {'Priority':<10}")
    print("  " + "-" * 56)
    for s in env.sensors:
        print(f"  S{s.node_id:<3} ({s.position[0]:6.1f}, {s.position[1]:6.1f})   {s.velocity:6.2f}    {s.energy:6.1f}     P{s.priority}")
    print()

    return env


# ============================================================
# STEP 2 - Simulate Mobile Sensor Nodes
# ============================================================

def run_step2(env):
    """Simulate sensor mobility over multiple timesteps and record trajectories."""

    print("=" * 65)
    print("  STEP 2 - Mobile Sensor Simulation")
    print("=" * 65)

    # Record trajectories for all sensors
    trajectories = {i: [env.sensors[i].position] for i in range(NUM_SENSORS)}
    energy_history = {i: [env.sensors[i].energy] for i in range(NUM_SENSORS)}

    # Also record UAV position (static in Phase 1, but tracked)
    uav_positions = [env.uav.position]

    for t in range(1, SIMULATION_STEPS):
        env.step()
        for i in range(NUM_SENSORS):
            trajectories[i].append(env.sensors[i].position)
            energy_history[i].append(env.sensors[i].energy)
        uav_positions.append(env.uav.position)

    print(f"  Simulated {SIMULATION_STEPS} timesteps of sensor mobility.")
    print()

    # Print movement summary for a few sensors
    print("  Sensor Movement Summary (Start -> End):")
    print(f"  {'ID':<4} {'Start Position':<22} {'End Position':<22} {'Distance':<12} {'Energy Drop':<12}")
    print("  " + "-" * 72)
    for i in range(min(10, NUM_SENSORS)):
        start = np.array(trajectories[i][0])
        end = np.array(trajectories[i][-1])
        dist = np.linalg.norm(end - start)
        e_drop = energy_history[i][0] - energy_history[i][-1]
        print(f"  S{i:<3} ({start[0]:6.1f}, {start[1]:6.1f})   ({end[0]:6.1f}, {end[1]:6.1f})   {dist:8.1f} m   {e_drop:8.2f}")
    if NUM_SENSORS > 10:
        print(f"  ... and {NUM_SENSORS - 10} more sensors")
    print()

    return trajectories, energy_history, uav_positions


# ============================================================
# STEP 3 - Generate Dynamic WSN Graph
# ============================================================

def run_step3(env, trajectories):
    """Build the WSN graph at multiple snapshots and track topology changes."""

    print("=" * 65)
    print("  STEP 3 - Dynamic WSN Graph")
    print("=" * 65)

    network = DynamicWSNGraph()

    # Reset environment and resimulate to capture graph snapshots
    env.reset()
    np.random.seed(42)
    env = SimulationEnvironment()

    graph_snapshots = {}
    topology_stats = []

    for t in range(SIMULATION_STEPS):
        if t > 0:
            env.step()

        graph = network.build_graph(env)

        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        num_components = nx.number_connected_components(graph)
        is_connected = nx.is_connected(graph)

        # Average link quality
        link_qualities = [d.get("link_quality", 0) for _, _, d in graph.edges(data=True)]
        avg_lq = np.mean(link_qualities) if link_qualities else 0.0

        topology_stats.append({
            "timestep": t,
            "nodes": num_nodes,
            "edges": num_edges,
            "components": num_components,
            "connected": is_connected,
            "avg_link_quality": avg_lq
        })

        if t in SNAPSHOT_TIMESTEPS:
            graph_snapshots[t] = {
                "graph": graph.copy(),
                "env_state": {
                    "sensors": [(s.position, s.energy, s.priority) for s in env.sensors],
                    "uav": env.uav.position,
                    "cs": env.command_station.position
                }
            }

    print(f"  Built dynamic graphs for {SIMULATION_STEPS} timesteps.")
    print()
    print("  Topology Statistics at Snapshot Timesteps:")
    print(f"  {'t':<6} {'Nodes':<8} {'Edges':<8} {'Components':<14} {'Connected':<12} {'Avg LQ':<10}")
    print("  " + "-" * 58)
    for t in SNAPSHOT_TIMESTEPS:
        s = topology_stats[t]
        print(f"  {s['timestep']:<6} {s['nodes']:<8} {s['edges']:<8} {s['components']:<14} {str(s['connected']):<12} {s['avg_link_quality']:.3f}")
    print()

    return network, graph_snapshots, topology_stats, env


# ============================================================
# STEP 4 - Communication Disruptions
# ============================================================

def run_step4(graph_snapshots, topology_stats):
    """Analyze the effect of communication disruptions on the network."""

    print("=" * 65)
    print("  STEP 4 - Communication Disruptions")
    print("=" * 65)

    print(f"  Disruption Zone   : X[{DISRUPTION_X_MIN}-{DISRUPTION_X_MAX}], Y[{DISRUPTION_Y_MIN}-{DISRUPTION_Y_MAX}]")
    print(f"  Link Quality Factor: 0.25 (75% degradation inside zone)")
    print()

    # Count how many edges pass through the disruption zone at each snapshot
    for t in SNAPSHOT_TIMESTEPS:
        g = graph_snapshots[t]["graph"]
        state = graph_snapshots[t]["env_state"]

        disrupted_edges = 0
        total_edges = g.number_of_edges()

        for u, v, d in g.edges(data=True):
            # Get positions
            if u.startswith("S"):
                idx = int(u[1:])
                pos_u = state["sensors"][idx][0]
            elif u == "UAV":
                pos_u = state["uav"]
            else:
                pos_u = state["cs"]

            if v.startswith("S"):
                idx = int(v[1:])
                pos_v = state["sensors"][idx][0]
            elif v == "UAV":
                pos_v = state["uav"]
            else:
                pos_v = state["cs"]

            midpoint = ((pos_u[0] + pos_v[0]) / 2, (pos_u[1] + pos_v[1]) / 2)
            if (DISRUPTION_X_MIN <= midpoint[0] <= DISRUPTION_X_MAX and
                    DISRUPTION_Y_MIN <= midpoint[1] <= DISRUPTION_Y_MAX):
                disrupted_edges += 1

        partitioned = not nx.is_connected(g)
        print(f"  t={t:2d}: Edges={total_edges:2d}, Disrupted Edges={disrupted_edges:2d}, "
              f"Components={nx.number_connected_components(g)}, "
              f"Network Partitioned={'YES [!]' if partitioned else 'No'}")

    print()

    # Track connectivity changes over time
    partitioned_steps = sum(1 for s in topology_stats if s["components"] > 1)
    connected_steps = sum(1 for s in topology_stats if s["components"] == 1)
    max_components = max(s["components"] for s in topology_stats)

    print(f"  Over {SIMULATION_STEPS} timesteps:")
    print(f"    Fully Connected Steps : {connected_steps}")
    print(f"    Partitioned Steps     : {partitioned_steps}")
    print(f"    Max Partitions (worst): {max_components} components")
    print()


# ============================================================
# COMPREHENSIVE PHASE 1 VISUALIZATION
# ============================================================

def generate_phase1_visualization(trajectories, energy_history, graph_snapshots,
                                   topology_stats, env):
    """Generate a comprehensive multi-panel visualization for Phase 1."""

    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#f8f9fa")

    # Use GridSpec for flexible layout: 3 rows
    gs = fig.add_gridspec(3, 4, hspace=0.35, wspace=0.35)

    # ----------------------------------------------------------
    # Panel 1 (top-left, spans 2 cols): Simulation Environment + Sensor Trajectories
    # ----------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0:2])

    # Surveillance corridor
    ax1.axvspan(CORRIDOR_X_MIN, CORRIDOR_X_MAX, alpha=0.08, color="green")
    ax1.add_patch(plt.Rectangle((CORRIDOR_X_MIN, CORRIDOR_Y_MIN),
                                 CORRIDOR_X_MAX - CORRIDOR_X_MIN,
                                 CORRIDOR_Y_MAX - CORRIDOR_Y_MIN,
                                 fill=False, edgecolor="green", linewidth=1.5,
                                 linestyle="--", label="Surveillance Corridor"))

    # Disruption zone
    if ENABLE_DISRUPTION:
        ax1.add_patch(plt.Rectangle((DISRUPTION_X_MIN, DISRUPTION_Y_MIN),
                                     DISRUPTION_X_MAX - DISRUPTION_X_MIN,
                                     DISRUPTION_Y_MAX - DISRUPTION_Y_MIN,
                                     facecolor="red", alpha=0.15, edgecolor="red",
                                     linewidth=2, linestyle="--",
                                     label="Disruption Zone"))

    # Draw sensor trajectories (colored by priority)
    priority_colors = {1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}
    for i in range(NUM_SENSORS):
        traj = trajectories[i]
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        prio = env.sensors[i].priority
        ax1.plot(xs, ys, "-", color=priority_colors[prio], alpha=0.35, linewidth=0.8)
        # Start dot
        ax1.scatter(xs[0], ys[0], color=priority_colors[prio], s=40,
                    edgecolors="black", linewidth=0.5, zorder=4)
        # End dot (larger)
        ax1.scatter(xs[-1], ys[-1], color=priority_colors[prio], s=80,
                    edgecolors="black", linewidth=0.8, zorder=5)
        ax1.annotate(f"S{i}", (xs[-1], ys[-1]), fontsize=6, ha="center",
                     va="bottom", fontweight="bold")

    # UAV
    ax1.scatter(*env.uav.position, marker="^", s=200, color="#e74c3c",
                edgecolors="black", linewidth=1.5, zorder=6, label="UAV")
    ax1.annotate("UAV", env.uav.position, fontsize=9, ha="center",
                 va="bottom", fontweight="bold", color="#e74c3c")

    # Command Station
    ax1.scatter(*env.command_station.position, marker="s", s=200, color="#27ae60",
                edgecolors="black", linewidth=1.5, zorder=6, label="Command Station")
    ax1.annotate("CS", env.command_station.position, fontsize=9, ha="center",
                 va="bottom", fontweight="bold", color="#27ae60")

    # Legend for priority colors
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#3498db",
               markersize=8, label='Priority 1 (Low)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#f39c12",
               markersize=8, label='Priority 2 (Medium)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor="#e74c3c",
               markersize=8, label='Priority 3 (Critical)'),
        Line2D([0], [0], marker='^', color='w', markerfacecolor="#e74c3c",
               markersize=10, label='UAV'),
        Line2D([0], [0], marker='s', color='w', markerfacecolor="#27ae60",
               markersize=10, label='Command Station'),
        mpatches.Patch(facecolor='green', alpha=0.15, label='Corridor'),
        mpatches.Patch(facecolor='red', alpha=0.2, label='Disruption Zone'),
    ]
    ax1.legend(handles=legend_elements, loc="upper right", fontsize=7,
               framealpha=0.9, ncol=2)

    ax1.set_xlim(0, AREA_WIDTH)
    ax1.set_ylim(0, AREA_HEIGHT)
    ax1.set_xlabel("X Position (m)", fontsize=10)
    ax1.set_ylabel("Y Position (m)", fontsize=10)
    ax1.set_title("Step 1 & 2: Simulation Environment + Sensor Trajectories",
                  fontsize=12, fontweight="bold", color="#2c3e50")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # ----------------------------------------------------------
    # Panel 2 (top-right, spans 2 cols): Energy Depletion Over Time
    # ----------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 2:4])

    timestep_range = range(SIMULATION_STEPS)
    for i in range(NUM_SENSORS):
        prio = env.sensors[i].priority
        ax2.plot(timestep_range, energy_history[i],
                 color=priority_colors[prio], alpha=0.5, linewidth=1.0)

    # Highlight a few sensors
    for i in [0, 5, 10, 15, 19]:
        prio = env.sensors[i].priority
        ax2.plot(timestep_range, energy_history[i],
                 color=priority_colors[prio], linewidth=2.0, alpha=0.9,
                 label=f"S{i} (P{prio})")

    ax2.set_xlabel("Timestep", fontsize=10)
    ax2.set_ylabel("Remaining Energy (%)", fontsize=10)
    ax2.set_title("Step 2: Sensor Energy Depletion Due to Mobility",
                  fontsize=12, fontweight="bold", color="#2c3e50")
    ax2.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax2.grid(True, linestyle="--", alpha=0.3)
    ax2.set_xlim(0, SIMULATION_STEPS - 1)

    # ----------------------------------------------------------
    # Panel 3 (middle row): 4 Graph Snapshots at different timesteps
    # ----------------------------------------------------------
    for panel_idx, t in enumerate(SNAPSHOT_TIMESTEPS):
        ax = fig.add_subplot(gs[1, panel_idx])

        snapshot = graph_snapshots[t]
        g = snapshot["graph"]
        state = snapshot["env_state"]

        # Build positions dict
        pos = {}
        node_colors = []
        node_sizes = []
        for i in range(NUM_SENSORS):
            s_name = f"S{i}"
            pos[s_name] = state["sensors"][i][0]
            prio = state["sensors"][i][2]
            node_colors.append(priority_colors[prio])
            node_sizes.append(60 + prio * 30)

        pos["UAV"] = state["uav"]
        pos["CS"] = state["cs"]

        # Corridor & disruption zone
        ax.axvspan(CORRIDOR_X_MIN, CORRIDOR_X_MAX, alpha=0.06, color="green")
        if ENABLE_DISRUPTION:
            ax.add_patch(plt.Rectangle(
                (DISRUPTION_X_MIN, DISRUPTION_Y_MIN),
                DISRUPTION_X_MAX - DISRUPTION_X_MIN,
                DISRUPTION_Y_MAX - DISRUPTION_Y_MIN,
                facecolor="red", alpha=0.12, edgecolor="red",
                linewidth=1, linestyle="--"))

        # Edge colors by link quality
        edge_colors = []
        edge_widths = []
        for u, v, d in g.edges(data=True):
            lq = d.get("link_quality", 0.5)
            if lq >= 0.7:
                edge_colors.append("#27ae60")
            elif lq >= 0.4:
                edge_colors.append("#f39c12")
            else:
                edge_colors.append("#e74c3c")
            edge_widths.append(0.5 + lq * 2.0)

        nx.draw_networkx_edges(g, pos, ax=ax, edge_color=edge_colors,
                               width=edge_widths, alpha=0.6)

        # Sensor nodes
        sensor_names = [f"S{i}" for i in range(NUM_SENSORS)]
        nx.draw_networkx_nodes(g, pos, nodelist=sensor_names, ax=ax,
                               node_size=node_sizes, node_color=node_colors,
                               edgecolors="black", linewidths=0.5)

        # UAV & CS
        nx.draw_networkx_nodes(g, pos, nodelist=["UAV"], ax=ax,
                               node_size=180, node_shape="^",
                               node_color="#e74c3c", edgecolors="black", linewidths=1)
        nx.draw_networkx_nodes(g, pos, nodelist=["CS"], ax=ax,
                               node_size=180, node_shape="s",
                               node_color="#27ae60", edgecolors="black", linewidths=1)

        num_comp = nx.number_connected_components(g)
        comp_color = "#27ae60" if num_comp == 1 else "#e74c3c"

        ax.set_xlim(0, AREA_WIDTH)
        ax.set_ylim(0, AREA_HEIGHT)
        ax.set_title(f"t = {t}   |  Edges: {g.number_of_edges()}  |  "
                     f"Components: {num_comp}",
                     fontsize=9, fontweight="bold", color=comp_color)
        ax.set_xlabel("X (m)", fontsize=8)
        ax.set_ylabel("Y (m)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.grid(True, linestyle="--", alpha=0.2)

    # ----------------------------------------------------------
    # Panel 4 (bottom-left, 2 cols): Topology Dynamics Over Time
    # ----------------------------------------------------------
    ax5 = fig.add_subplot(gs[2, 0:2])

    timesteps = [s["timestep"] for s in topology_stats]
    edges_over_time = [s["edges"] for s in topology_stats]
    components_over_time = [s["components"] for s in topology_stats]
    lq_over_time = [s["avg_link_quality"] for s in topology_stats]

    ax5.plot(timesteps, edges_over_time, "-", color="#2980b9", linewidth=2,
             label="Active Edges", alpha=0.9)
    ax5.fill_between(timesteps, edges_over_time, alpha=0.1, color="#2980b9")

    ax5_twin = ax5.twinx()
    ax5_twin.plot(timesteps, components_over_time, "-", color="#e74c3c",
                  linewidth=2, label="Connected Components")
    ax5_twin.fill_between(timesteps, components_over_time, alpha=0.1, color="#e74c3c")
    ax5_twin.set_ylabel("Connected Components", fontsize=10, color="#e74c3c")
    ax5_twin.tick_params(axis="y", labelcolor="#e74c3c")

    # Mark snapshot timesteps
    for t in SNAPSHOT_TIMESTEPS:
        ax5.axvline(t, color="gray", linestyle=":", alpha=0.5)
        ax5.text(t, max(edges_over_time) + 1, f"t={t}", fontsize=7,
                 ha="center", color="gray")

    ax5.set_xlabel("Timestep", fontsize=10)
    ax5.set_ylabel("Number of Active Edges", fontsize=10, color="#2980b9")
    ax5.tick_params(axis="y", labelcolor="#2980b9")
    ax5.set_title("Step 3: Dynamic Topology Changes Over Time",
                  fontsize=12, fontweight="bold", color="#2c3e50")
    ax5.set_xlim(0, SIMULATION_STEPS - 1)
    ax5.grid(True, linestyle="--", alpha=0.3)

    # Combined legend
    lines1, labels1 = ax5.get_legend_handles_labels()
    lines2, labels2 = ax5_twin.get_legend_handles_labels()
    ax5.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=9)

    # ----------------------------------------------------------
    # Panel 5 (bottom-right, 2 cols): Disruption Impact Analysis
    # ----------------------------------------------------------
    ax6 = fig.add_subplot(gs[2, 2:4])

    ax6.plot(timesteps, lq_over_time, "-", color="#8e44ad", linewidth=2,
             label="Avg Link Quality", alpha=0.9)
    ax6.fill_between(timesteps, lq_over_time, alpha=0.1, color="#8e44ad")

    # Highlight partitioned timesteps
    partitioned_ts = [s["timestep"] for s in topology_stats if s["components"] > 1]
    connected_ts = [s["timestep"] for s in topology_stats if s["components"] == 1]

    if partitioned_ts:
        ax6.scatter(partitioned_ts,
                    [lq_over_time[t] for t in partitioned_ts],
                    color="#e74c3c", s=15, zorder=4, label="Partitioned", alpha=0.7)
    if connected_ts:
        ax6.scatter(connected_ts,
                    [lq_over_time[t] for t in connected_ts],
                    color="#27ae60", s=8, zorder=3, label="Connected", alpha=0.4)

    ax6.axhline(0.3, color="#e74c3c", linestyle="--", alpha=0.5,
                label="Link Quality Threshold (0.3)")

    # Summary stats box
    stats_text = (
        f"Phase 1 Summary\n"
        f"---------------------\n"
        f"Sensors Deployed : {NUM_SENSORS}\n"
        f"Simulation Steps : {SIMULATION_STEPS}\n"
        f"Sensor Comm Range: {SENSOR_COMMUNICATION_RANGE}m\n"
        f"UAV Comm Range   : {UAV_COMMUNICATION_RANGE}m\n"
        f"Fully Connected  : {len(connected_ts)} steps\n"
        f"Partitioned      : {len(partitioned_ts)} steps\n"
        f"Avg Link Quality : {np.mean(lq_over_time):.3f}"
    )
    ax6.text(0.98, 0.97, stats_text, transform=ax6.transAxes,
             verticalalignment="top", horizontalalignment="right",
             fontsize=8, family="monospace",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                       edgecolor="#bdc3c7", alpha=0.95))

    ax6.set_xlabel("Timestep", fontsize=10)
    ax6.set_ylabel("Average Link Quality", fontsize=10)
    ax6.set_title("Step 4: Communication Disruption Impact",
                  fontsize=12, fontweight="bold", color="#2c3e50")
    ax6.set_xlim(0, SIMULATION_STEPS - 1)
    ax6.set_ylim(0, 1.0)
    ax6.legend(loc="lower left", fontsize=8, framealpha=0.9)
    ax6.grid(True, linestyle="--", alpha=0.3)

    # ----------------------------------------------------------
    # Main Title
    # ----------------------------------------------------------
    fig.suptitle(
        "Phase 1: Environment & Data Pipeline - Complete Simulation Output\n"
        "UAV-Assisted Mobile WSN in Mountain-Pass Surveillance Corridor",
        fontsize=16, fontweight="bold", color="#2c3e50", y=0.99
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    output_file = "phase1_complete_output.png"
    plt.savefig(output_file, dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)

    print(f"  Visualization saved -> {output_file}")
    return output_file


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    print()
    print("+" + "=" * 63 + "+")
    print("|   PHASE 1: Environment & Data Pipeline - Complete Demo        |")
    print("|   Steps 1-4: Environment -> Sensors -> Graph -> Disruptions   |")
    print("+" + "=" * 63 + "+")
    print()

    # Step 1: Create environment
    env = run_step1()

    # Step 2: Simulate mobile sensors
    trajectories, energy_history, uav_positions = run_step2(env)

    # Step 3: Build dynamic WSN graph (re-creates env from same seed)
    network, graph_snapshots, topology_stats, env = run_step3(env, trajectories)

    # Recollect trajectories from the fresh env for the visualization
    env2 = SimulationEnvironment()
    np.random.seed(42)
    env2 = SimulationEnvironment()
    traj2 = {i: [env2.sensors[i].position] for i in range(NUM_SENSORS)}
    ehist2 = {i: [env2.sensors[i].energy] for i in range(NUM_SENSORS)}
    for t in range(1, SIMULATION_STEPS):
        env2.step()
        for i in range(NUM_SENSORS):
            traj2[i].append(env2.sensors[i].position)
            ehist2[i].append(env2.sensors[i].energy)

    # Step 4: Analyze communication disruptions
    run_step4(graph_snapshots, topology_stats)

    # Generate comprehensive visualization
    print("=" * 65)
    print("  VISUALIZATION - Generating Phase 1 Output")
    print("=" * 65)
    output = generate_phase1_visualization(traj2, ehist2, graph_snapshots,
                                            topology_stats, env2)

    print()
    print("+" + "=" * 63 + "+")
    print("|   PHASE 1 COMPLETE                                           |")
    print("|                                                               |")
    print("|   Generated Files:                                            |")
    print("|     - phase1_complete_output.png                              |")
    print("|                                                               |")
    print("|   What was demonstrated:                                      |")
    print("|     Step 1 -> Mountain-pass surveillance environment           |")
    print("|     Step 2 -> 20 mobile sensors with energy depletion          |")
    print("|     Step 3 -> Dynamic WSN graph topology over 50 timesteps     |")
    print("|     Step 4 -> Terrain communication disruptions & partitions   |")
    print("+" + "=" * 63 + "+")
    print()
