import numpy as np

from environment import SimulationEnvironment
from network import DynamicWSNGraph
from visualization import visualize_step


# --------------------------------------------------
# Make the simulation reproducible
# --------------------------------------------------

np.random.seed(42)


# --------------------------------------------------
# Create simulation environment
# --------------------------------------------------

environment = SimulationEnvironment()

network = DynamicWSNGraph()


# --------------------------------------------------
# Timestep 0
# --------------------------------------------------

graph = network.build_graph(environment)

print(
    f"Timestep: {environment.current_step} | "
    f"Nodes: {graph.number_of_nodes()} | "
    f"Edges: {graph.number_of_edges()} | "
    f"Components: {__import__('networkx').number_connected_components(graph)}"
)


visualize_step(
    environment,
    graph,
    "timestep_0_disruption.png"
)


# --------------------------------------------------
# Move simulation to timestep 20
# --------------------------------------------------

for step in range(20):

    environment.step()


# --------------------------------------------------
# Timestep 20
# --------------------------------------------------

graph = network.build_graph(environment)

print(
    f"Timestep: {environment.current_step} | "
    f"Nodes: {graph.number_of_nodes()} | "
    f"Edges: {graph.number_of_edges()} | "
    f"Components: {__import__('networkx').number_connected_components(graph)}"
)


visualize_step(
    environment,
    graph,
    "timestep_20_disruption.png"
)


print()
print("Step 4 visualization completed.")
print("Generated:")
print(" - timestep_0_disruption.png")
print(" - timestep_20_disruption.png")