import matplotlib.pyplot as plt
import networkx as nx

from config import *


def visualize_graph(environment, graph):

    fig, ax = plt.subplots(figsize=(12, 9))

    # Node positions
    positions = {}

    for sensor in environment.sensors:

        positions[f"S{sensor.node_id}"] = sensor.position

    positions["UAV"] = environment.uav.position

    positions["CS"] = environment.command_station.position

    # Draw surveillance corridor

    ax.axvspan(
        CORRIDOR_X_MIN,
        CORRIDOR_X_MAX,
        alpha=0.15,
        label="Surveillance Corridor"
    )

    # Separate nodes by type

    sensor_nodes = [
        node for node, data in graph.nodes(data=True)
        if data["node_type"] == "sensor"
    ]

    uav_nodes = ["UAV"]

    command_nodes = ["CS"]

    # Draw communication links

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        edge_color="gray",
        width=1.5,
        alpha=0.7
    )

    # Draw sensors

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=sensor_nodes,
        node_size=500,
        node_color="skyblue",
        node_shape="o",
        edgecolors="black",
        ax=ax
    )

    # Draw UAV

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=uav_nodes,
        node_size=900,
        node_color="orange",
        node_shape="^",
        edgecolors="black",
        ax=ax
    )

    # Draw Command Station

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=command_nodes,
        node_size=900,
        node_color="red",
        node_shape="s",
        edgecolors="black",
        ax=ax
    )

    # Node labels

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9,
        font_weight="bold",
        ax=ax
    )

    # Graph information

    ax.set_title(
        f"Dynamic WSN Communication Graph - "
        f"Timestep {environment.current_step}",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("X Position (m)")
    ax.set_ylabel("Y Position (m)")

    ax.set_xlim(0, environment.width)
    ax.set_ylim(0, environment.height)

    ax.grid(alpha=0.3)

    ax.legend()

    plt.tight_layout()

    # Save screenshot automatically

    plt.savefig(
        "step3_dynamic_wsn_graph.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.show()