import matplotlib.pyplot as plt
import networkx as nx

from config import *


def visualize_step(environment, graph, filename):

    fig, ax = plt.subplots(figsize=(12, 9))

    # --------------------------------------------------
    # Get node positions
    # --------------------------------------------------

    positions = {}

    for sensor in environment.sensors:

        positions[f"S{sensor.node_id}"] = sensor.position

    positions["UAV"] = environment.uav.position

    positions["CS"] = environment.command_station.position


    # --------------------------------------------------
    # Draw surveillance corridor
    # --------------------------------------------------

    ax.axvspan(
        CORRIDOR_X_MIN,
        CORRIDOR_X_MAX,
        alpha=0.10,
        label="Surveillance Corridor"
    )


    # --------------------------------------------------
    # Draw communication disruption zone
    # --------------------------------------------------

    if ENABLE_DISRUPTION:

        ax.fill_between(
            [DISRUPTION_X_MIN, DISRUPTION_X_MAX],
            DISRUPTION_Y_MIN,
            DISRUPTION_Y_MAX,
            alpha=0.25,
            label="Communication Disruption Zone"
        )

        ax.plot(
            [
                DISRUPTION_X_MIN,
                DISRUPTION_X_MAX,
                DISRUPTION_X_MAX,
                DISRUPTION_X_MIN,
                DISRUPTION_X_MIN
            ],
            [
                DISRUPTION_Y_MIN,
                DISRUPTION_Y_MIN,
                DISRUPTION_Y_MAX,
                DISRUPTION_Y_MAX,
                DISRUPTION_Y_MIN
            ],
            linestyle="--",
            linewidth=2
        )


    # --------------------------------------------------
    # Separate nodes
    # --------------------------------------------------

    sensor_nodes = [
        node
        for node, data in graph.nodes(data=True)
        if data["node_type"] == "sensor"
    ]

    uav_nodes = ["UAV"]

    command_nodes = ["CS"]


    # --------------------------------------------------
    # Draw communication edges
    # --------------------------------------------------

    nx.draw_networkx_edges(
        graph,
        positions,
        ax=ax,
        width=1.5,
        alpha=0.7
    )


    # --------------------------------------------------
    # Draw sensor nodes
    # --------------------------------------------------

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=sensor_nodes,
        node_size=500,
        node_shape="o",
        edgecolors="black",
        ax=ax
    )


    # --------------------------------------------------
    # Draw UAV
    # --------------------------------------------------

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=uav_nodes,
        node_size=900,
        node_shape="^",
        edgecolors="black",
        ax=ax
    )


    # --------------------------------------------------
    # Draw Command Station
    # --------------------------------------------------

    nx.draw_networkx_nodes(
        graph,
        positions,
        nodelist=command_nodes,
        node_size=900,
        node_shape="s",
        edgecolors="black",
        ax=ax
    )


    # --------------------------------------------------
    # Draw labels
    # --------------------------------------------------

    nx.draw_networkx_labels(
        graph,
        positions,
        font_size=9,
        font_weight="bold",
        ax=ax
    )


    # --------------------------------------------------
    # Calculate graph statistics
    # --------------------------------------------------

    number_of_nodes = graph.number_of_nodes()

    number_of_edges = graph.number_of_edges()

    number_of_components = nx.number_connected_components(
        graph
    )

    connected = nx.is_connected(graph)


    # --------------------------------------------------
    # Graph statistics box
    # --------------------------------------------------

    statistics = (
        f"Nodes: {number_of_nodes}\n"
        f"Communication Links: {number_of_edges}\n"
        f"Network Components: {number_of_components}\n"
        f"Connected: {connected}"
    )

    ax.text(
        0.02,
        0.98,
        statistics,
        transform=ax.transAxes,
        verticalalignment="top",
        fontsize=11,
        bbox=dict(
            boxstyle="round",
            facecolor="white",
            alpha=0.85
        )
    )


    # --------------------------------------------------
    # Title
    # --------------------------------------------------

    ax.set_title(
        f"Step 4 - Communication Disruption\n"
        f"Timestep {environment.current_step}",
        fontsize=16,
        fontweight="bold"
    )


    # --------------------------------------------------
    # Axis labels
    # --------------------------------------------------

    ax.set_xlabel(
        "X Position (m)",
        fontsize=12
    )

    ax.set_ylabel(
        "Y Position (m)",
        fontsize=12
    )


    # --------------------------------------------------
    # Axis limits
    # --------------------------------------------------

    ax.set_xlim(
        0,
        environment.width
    )

    ax.set_ylim(
        0,
        environment.height
    )


    # --------------------------------------------------
    # Grid and legend
    # --------------------------------------------------

    ax.grid(
        alpha=0.3
    )

    ax.legend(
        loc="upper right"
    )


    plt.tight_layout()


    # --------------------------------------------------
    # Save screenshot
    # --------------------------------------------------

    plt.savefig(
        filename,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close(fig)