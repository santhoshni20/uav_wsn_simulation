"""
=============================================================================
Phase 1: Live Animated UAV-WSN Simulation
=============================================================================
Demonstrates real-time mobile sensor dynamics, dynamic link formation,
and terrain-induced RF communication disruptions in the mountain pass corridor.

Controls:
  - Spacebar : Pause / Resume
  - R        : Reset simulation
  - Q / Esc  : Close window

Run:
  python live_simulation.py
=============================================================================
"""

import sys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Rectangle
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
    TOTAL_STEPS
)
from environment import SimulationEnvironment
from network import DynamicWSNGraph


class LiveWSNSimulation:

    def __init__(self, total_steps=100, interval=120):
        self.total_steps = total_steps
        self.interval = interval
        self.is_paused = False

        # Create simulation and network models
        np.random.seed(42)
        self.env = SimulationEnvironment()
        self.network = DynamicWSNGraph()

        # Priority color mapping
        self.priority_colors = {1: "#3498db", 2: "#f39c12", 3: "#e74c3c"}

        # Setup figure
        self.fig, self.ax = plt.subplots(figsize=(13, 9.5))
        self.fig.canvas.manager.set_window_title(
            "Phase 1: Real-Time UAV-WSN Simulation (Mountain-Pass Corridor)"
        )
        self.fig.patch.set_facecolor("#f8f9fa")

        # Connect keyboard events for interactive control
        self.fig.canvas.mpl_connect("key_press_event", self.on_key_press)

        self.setup_static_elements()
        self.setup_dynamic_elements()

    def setup_static_elements(self):
        """Draw the terrain, corridor, and disruption zone."""
        self.ax.set_facecolor("#ffffff")

        # Surveillance Corridor
        corridor = Rectangle(
            (CORRIDOR_X_MIN, CORRIDOR_Y_MIN),
            CORRIDOR_X_MAX - CORRIDOR_X_MIN,
            CORRIDOR_Y_MAX - CORRIDOR_Y_MIN,
            facecolor="#27ae60",
            alpha=0.08,
            edgecolor="#27ae60",
            linewidth=2,
            linestyle="--",
            label="Mountain Surveillance Corridor"
        )
        self.ax.add_patch(corridor)

        # Terrain RF Disruption Zone
        if ENABLE_DISRUPTION:
            disruption = Rectangle(
                (DISRUPTION_X_MIN, DISRUPTION_Y_MIN),
                DISRUPTION_X_MAX - DISRUPTION_X_MIN,
                DISRUPTION_Y_MAX - DISRUPTION_Y_MIN,
                facecolor="#e74c3c",
                alpha=0.18,
                edgecolor="#c0392b",
                linewidth=2,
                linestyle=":",
                hatch="//",
                label="Deep RF Disruption Zone (75% Link Drop)"
            )
            self.ax.add_patch(disruption)

        # Axis styling
        self.ax.set_xlim(0, AREA_WIDTH)
        self.ax.set_ylim(0, AREA_HEIGHT)
        self.ax.set_xlabel("Corridor X Position (meters)", fontsize=11, fontweight="bold")
        self.ax.set_ylabel("Corridor Y Position (meters)", fontsize=11, fontweight="bold")
        self.ax.grid(True, linestyle="--", alpha=0.3)

        # Legend
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#3498db", markersize=9, label='Sensor Priority 1 (Low)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#f39c12", markersize=9, label='Sensor Priority 2 (Medium)'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor="#e74c3c", markersize=9, label='Sensor Priority 3 (Critical)'),
            Line2D([0], [0], marker='^', color='w', markerfacecolor="#e74c3c", markersize=11, label='UAV Aerial Relay'),
            Line2D([0], [0], marker='s', color='w', markerfacecolor="#27ae60", markersize=11, label='Command Station'),
            Line2D([0], [0], color="#27ae60", lw=2, label='Active Link (Normal)'),
            Line2D([0], [0], color="#e74c3c", lw=2, linestyle="--", label='Degraded Link (In Disruption)'),
        ]
        self.ax.legend(handles=legend_elements, loc="upper right", fontsize=8.5, framealpha=0.92)

    def setup_dynamic_elements(self):
        """Initialize dynamic artists for nodes, links, and HUD."""
        # Sensor scatter
        sensor_xs = [s.position[0] for s in self.env.sensors]
        sensor_ys = [s.position[1] for s in self.env.sensors]
        sensor_colors = [self.priority_colors[s.priority] for s in self.env.sensors]
        self.sensor_scatter = self.ax.scatter(
            sensor_xs, sensor_ys,
            c=sensor_colors, s=120, edgecolors="black", linewidths=1.2, zorder=5
        )

        # Sensor labels
        self.sensor_labels = []
        for s in self.env.sensors:
            lbl = self.ax.text(
                s.position[0], s.position[1] + 12, f"S{s.node_id}",
                fontsize=7.5, fontweight="bold", ha="center", va="bottom", zorder=6
            )
            self.sensor_labels.append(lbl)

        # UAV node
        self.uav_scatter = self.ax.scatter(
            [self.env.uav.position[0]], [self.env.uav.position[1]],
            marker="^", s=280, color="#e74c3c", edgecolors="black", linewidths=1.5, zorder=7
        )
        self.uav_label = self.ax.text(
            self.env.uav.position[0], self.env.uav.position[1] + 16, "UAV Relay",
            fontsize=8.5, fontweight="bold", ha="center", color="#c0392b", zorder=8
        )

        # Command station
        self.cs_scatter = self.ax.scatter(
            [self.env.command_station.position[0]], [self.env.command_station.position[1]],
            marker="s", s=280, color="#27ae60", edgecolors="black", linewidths=1.5, zorder=7
        )
        self.cs_label = self.ax.text(
            self.env.command_station.position[0], self.env.command_station.position[1] - 22, "Base Command Station",
            fontsize=8.5, fontweight="bold", ha="center", color="#1e8449", zorder=8
        )

        # Edges container (list of Line2D)
        self.edge_lines = []

        # HUD Box
        self.hud_text = self.ax.text(
            0.02, 0.98, "",
            transform=self.ax.transAxes,
            verticalalignment="top",
            fontsize=9.5,
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.6", facecolor="white", edgecolor="#bdc3c7", alpha=0.92),
            zorder=10
        )

        # Controls hint text at bottom
        self.ax.text(
            0.5, 0.015,
            "[Spacebar: Pause / Resume]   [R: Restart]   [Q: Quit]",
            transform=self.ax.transAxes,
            horizontalalignment="center",
            fontsize=8.5,
            color="#7f8c8d",
            bbox=dict(boxstyle="square,pad=0.2", facecolor="#ffffff", edgecolor="none", alpha=0.8)
        )

    def update_frame(self, frame):
        """Update sensor positions, graph links, and stats for one timestep."""
        if self.is_paused:
            return

        if frame > 0:
            self.env.step()

        # Build dynamic graph
        graph = self.network.build_graph(self.env)

        # Update sensor positions
        coords = np.array([s.position for s in self.env.sensors])
        self.sensor_scatter.set_offsets(coords)

        # Update labels
        for idx, s in enumerate(self.env.sensors):
            self.sensor_labels[idx].set_position((s.position[0], s.position[1] + 12))

        # Remove old edge lines
        for line in self.edge_lines:
            line.remove()
        self.edge_lines.clear()

        # Draw new edges
        pos_dict = {f"S{s.node_id}": s.position for s in self.env.sensors}
        pos_dict["UAV"] = self.env.uav.position
        pos_dict["CS"] = self.env.command_station.position

        total_lq = 0.0
        disrupted_count = 0

        for u, v, d in graph.edges(data=True):
            p1 = pos_dict[u]
            p2 = pos_dict[v]
            lq = d.get("link_quality", 0.5)
            total_lq += lq

            # Check if edge traverses the disruption zone
            midpoint = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
            in_disruption = (
                ENABLE_DISRUPTION and
                DISRUPTION_X_MIN <= midpoint[0] <= DISRUPTION_X_MAX and
                DISRUPTION_Y_MIN <= midpoint[1] <= DISRUPTION_Y_MAX
            )

            if in_disruption:
                color = "#e74c3c"
                ls = "--"
                lw = 1.2
                disrupted_count += 1
            else:
                color = "#27ae60" if lq >= 0.5 else "#f39c12"
                ls = "-"
                lw = 0.8 + lq * 1.8

            line = self.ax.plot(
                [p1[0], p2[0]], [p1[1], p2[1]],
                color=color, linestyle=ls, linewidth=lw, alpha=0.65, zorder=3
            )[0]
            self.edge_lines.append(line)

        # Calculate statistics
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        num_components = nx.number_connected_components(graph)
        is_connected = nx.is_connected(graph)
        avg_lq = (total_lq / num_edges) if num_edges > 0 else 0.0

        status_flag = "[CONNECTED]" if is_connected else f"[PARTITIONED: {num_components} Components]"
        status_color = "#27ae60" if is_connected else "#e74c3c"

        hud = (
            f"Step: {self.env.current_step:02d} / {self.total_steps}  |  Nodes: {num_nodes}\n"
            f"Active Links        : {num_edges}\n"
            f"Disrupted Links     : {disrupted_count}\n"
            f"Network Status      : {status_flag}\n"
            f"Avg Channel Quality : {avg_lq:.3f}\n"
            f"UAV Battery Level   : {self.env.uav.energy:.1f}%"
        )
        self.hud_text.set_text(hud)

        self.ax.set_title(
            f"Phase 1: Mountain-Pass Surveillance Corridor Simulation  (Timestep: {self.env.current_step})",
            fontsize=13, fontweight="bold", color="#2c3e50"
        )

    def on_key_press(self, event):
        """Handle keyboard interactions."""
        if event.key == " ":
            self.is_paused = not self.is_paused
        elif event.key in ("r", "R"):
            self.env.reset()
            self.is_paused = False
        elif event.key in ("q", "Q", "escape"):
            plt.close(self.fig)

    def run(self):
        """Start the real-time simulation loop."""
        self.anim = animation.FuncAnimation(
            self.fig,
            self.update_frame,
            frames=self.total_steps,
            interval=self.interval,
            repeat=True
        )
        plt.show()


def run_live_simulation(total_steps=100, interval=100):
    sim = LiveWSNSimulation(total_steps=total_steps, interval=interval)
    sim.run()


if __name__ == "__main__":
    print()
    print("=" * 65)
    print("  Launching Live Animated UAV-WSN Simulation...")
    print("  Controls: Spacebar to Pause/Resume, R to Reset, Q to Quit")
    print("=" * 65)
    print()
    run_live_simulation()
