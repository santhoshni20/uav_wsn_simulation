from environment import SimulationEnvironment
from network import DynamicWSNGraph
from visualization import visualize_graph


environment = SimulationEnvironment()

network = DynamicWSNGraph()


graph = network.build_graph(environment)

print(
    f"Step {environment.current_step} | "
    f"Nodes: {graph.number_of_nodes()} | "
    f"Edges: {graph.number_of_edges()}"
)

visualize_graph(
    environment,
    graph
)