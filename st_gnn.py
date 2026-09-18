import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from config import (
    AREA_WIDTH,
    AREA_HEIGHT,
    SENSOR_MAX_SPEED,
    NUM_SENSORS,
    MAX_PRIORITY
)
from environment import SimulationEnvironment
from network import DynamicWSNGraph


# -------------------------------------------------------------
# Graph snapshot feature extraction
# -------------------------------------------------------------

def extract_node_features_and_adj(environment, graph):
    """
    Extracts normalized node feature matrix X and normalized weighted adjacency A.
    Node order: S0..S19, UAV, CS (Total = NUM_SENSORS + 2 = 22)
    Features (dim=8):
      [norm_x, norm_y, norm_vx, norm_vy, norm_energy, norm_priority, is_uav, is_cs]
    """
    node_keys = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]
    num_nodes = len(node_keys)
    features = []

    # Sensor features
    for i in range(NUM_SENSORS):
        sensor = environment.sensors[i]
        vx = (sensor.velocity * np.cos(sensor.direction)) / SENSOR_MAX_SPEED
        vy = (sensor.velocity * np.sin(sensor.direction)) / SENSOR_MAX_SPEED
        features.append([
            sensor.position[0] / AREA_WIDTH,
            sensor.position[1] / AREA_HEIGHT,
            vx,
            vy,
            sensor.energy / 100.0,
            sensor.priority / MAX_PRIORITY,
            0.0,  # is_uav
            0.0   # is_cs
        ])

    # UAV features
    uav = environment.uav
    features.append([
        uav.position[0] / AREA_WIDTH,
        uav.position[1] / AREA_HEIGHT,
        0.0,
        0.0,
        uav.energy / 100.0,
        1.0,  # high priority
        1.0,  # is_uav
        0.0   # is_cs
    ])

    # Command Station features
    cs = environment.command_station
    features.append([
        cs.position[0] / AREA_WIDTH,
        cs.position[1] / AREA_HEIGHT,
        0.0,
        0.0,
        1.0,  # infinite energy
        1.0,  # high priority
        0.0,  # is_uav
        1.0   # is_cs
    ])

    X = np.array(features, dtype=np.float32)

    # Weighted Adjacency Matrix
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    key_to_idx = {k: idx for idx, k in enumerate(node_keys)}

    for u, v, data in graph.edges(data=True):
        if u in key_to_idx and v in key_to_idx:
            idx_u = key_to_idx[u]
            idx_v = key_to_idx[v]
            w = float(data.get("link_quality", 1.0))
            A[idx_u, idx_v] = w
            A[idx_v, idx_u] = w

    # Add self-loops
    A_tilde = A + np.eye(num_nodes, dtype=np.float32)
    degrees = np.sum(A_tilde, axis=1)
    d_inv_sqrt = np.zeros_like(degrees, dtype=np.float32)
    pos_mask = degrees > 0
    d_inv_sqrt[pos_mask] = np.power(degrees[pos_mask], -0.5)
    D_mat = np.diag(d_inv_sqrt)
    A_norm = D_mat @ A_tilde @ D_mat

    return torch.tensor(X, dtype=torch.float32), torch.tensor(A_norm, dtype=torch.float32)


# -------------------------------------------------------------
# Spatial Graph Convolution Layer
# -------------------------------------------------------------

class GraphConvolution(nn.Module):
    def __init__(self, in_features, out_features):
        super(GraphConvolution, self).__init__()
        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        self.bias = nn.Parameter(torch.FloatTensor(out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        nn.init.zeros_(self.bias)

    def forward(self, x, adj):
        """
        x: (batch, num_nodes, in_features) or (num_nodes, in_features)
        adj: (batch, num_nodes, num_nodes) or (num_nodes, num_nodes)
        """
        if x.dim() == 2:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(adj, support) + self.bias
        else:
            support = torch.matmul(x, self.weight)
            output = torch.matmul(adj, support) + self.bias
        return F.relu(output)


# -------------------------------------------------------------
# Spatio-Temporal Graph Neural Network (ST-GNN)
# -------------------------------------------------------------

class STGNN(nn.Module):
    """
    Spatio-Temporal Graph Neural Network combining Spatial Graph Convolutions
    with a Temporal Gated Recurrent Unit (GRU) to model dynamic WSN topologies.
    """
    def __init__(self, in_features=8, spatial_dim=32, temporal_dim=64, out_dim=64):
        super(STGNN, self).__init__()
        self.in_features = in_features
        self.spatial_dim = spatial_dim
        self.temporal_dim = temporal_dim
        self.out_dim = out_dim

        # Two-layer Spatial GCN
        self.gcn1 = GraphConvolution(in_features, spatial_dim)
        self.gcn2 = GraphConvolution(spatial_dim, spatial_dim)

        # Temporal GRU operating over time window per node
        self.temporal_gru = nn.GRU(
            input_size=spatial_dim,
            hidden_size=temporal_dim,
            batch_first=True
        )

        # Output projections
        self.node_proj = nn.Linear(temporal_dim, out_dim)
        self.graph_proj = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim)
        )

    def forward(self, x_seq, adj_seq):
        """
        x_seq: (seq_len, num_nodes, in_features) or (batch, seq_len, num_nodes, in_features)
        adj_seq: (seq_len, num_nodes, num_nodes) or (batch, seq_len, num_nodes, num_nodes)

        Returns:
          node_embeddings: (batch, num_nodes, out_dim)
          graph_embedding: (batch, out_dim)
        """
        has_batch = (x_seq.dim() == 4)
        if not has_batch:
            x_seq = x_seq.unsqueeze(0)
            adj_seq = adj_seq.unsqueeze(0)

        batch_size, seq_len, num_nodes, _ = x_seq.shape

        # 1. Spatial GCN across all timesteps
        spatial_outputs = []
        for t in range(seq_len):
            xt = x_seq[:, t, :, :]
            adjt = adj_seq[:, t, :, :]
            h1 = self.gcn1(xt, adjt)
            h2 = self.gcn2(h1, adjt)
            spatial_outputs.append(h2)

        # (batch, seq_len, num_nodes, spatial_dim)
        spatial_stack = torch.stack(spatial_outputs, dim=1)

        # 2. Temporal modeling per node
        # Reshape to (batch * num_nodes, seq_len, spatial_dim)
        temporal_in = spatial_stack.permute(0, 2, 1, 3).reshape(
            batch_size * num_nodes, seq_len, self.spatial_dim
        )
        gru_out, _ = self.temporal_gru(temporal_in)
        # Take the final temporal state for each node: (batch * num_nodes, temporal_dim)
        node_temporal = gru_out[:, -1, :]

        # 3. Reshape and project
        node_embeddings = self.node_proj(node_temporal).reshape(
            batch_size, num_nodes, self.out_dim
        )

        # Global graph embedding by mean pooling + MLP
        graph_embedding = self.graph_proj(torch.mean(node_embeddings, dim=1))

        if not has_batch:
            return node_embeddings.squeeze(0), graph_embedding.squeeze(0)

        return node_embeddings, graph_embedding


# -------------------------------------------------------------
# Step 5 Verification & Visualization
# -------------------------------------------------------------

def run_step5():
    print("=" * 60)
    print("Step 5 - Implementing ST-GNN for Dynamic WSN")
    print("=" * 60)

    np.random.seed(42)
    torch.manual_seed(42)

    env = SimulationEnvironment()
    network = DynamicWSNGraph()
    model = STGNN(in_features=8, spatial_dim=32, temporal_dim=64, out_dim=64)
    model.eval()

    # Collect 5 consecutive timesteps of graph snapshots
    time_window = 5
    x_history = []
    adj_history = []
    connectivity_stats = []

    print(f"Simulating {time_window} timesteps to construct spatio-temporal window...")
    for t in range(time_window):
        if t > 0:
            env.step()
        g = network.build_graph(env)
        x_t, adj_t = extract_node_features_and_adj(env, g)
        x_history.append(x_t)
        adj_history.append(adj_t)
        num_components = nx.number_connected_components(g)
        num_edges = g.number_of_edges()
        connectivity_stats.append((t, g.number_of_nodes(), num_edges, num_components))
        print(f"  t={t}: Nodes={g.number_of_nodes()}, Edges={num_edges}, Connected Components={num_components}")

    x_tensor = torch.stack(x_history, dim=0)       # (T, N, F)
    adj_tensor = torch.stack(adj_history, dim=0)   # (T, N, N)

    print("\nForward pass through ST-GNN...")
    with torch.no_grad():
        node_emb, graph_emb = model(x_tensor, adj_tensor)

    print(f"  Input sequence shape:   X={x_tensor.shape}, Adj={adj_tensor.shape}")
    print(f"  Node embeddings shape:  {node_emb.shape}  (22 nodes x 64 latent dims)")
    print(f"  Graph embedding shape: {graph_emb.shape} (64 global latent dims)")
    print(f"  Graph embedding norm:  {graph_emb.norm().item():.4f}")

    # -------------------------------------------------------------
    # Generate Step 5 Visualization
    # -------------------------------------------------------------
    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 3)

    # 1. Top left: Snapshot 0 graph
    ax1 = fig.add_subplot(gs[0, 0])
    g0 = network.build_graph(env)
    pos = {f"S{s.node_id}": s.position for s in env.sensors}
    pos["UAV"] = env.uav.position
    pos["CS"] = env.command_station.position
    nx.draw_networkx_edges(g0, pos, ax=ax1, alpha=0.5, edge_color="gray")
    nx.draw_networkx_nodes(g0, pos, nodelist=[f"S{i}" for i in range(NUM_SENSORS)], ax=ax1, node_size=120, node_color="#4575b4")
    nx.draw_networkx_nodes(g0, pos, nodelist=["UAV"], ax=ax1, node_size=250, node_shape="^", node_color="#d73027")
    nx.draw_networkx_nodes(g0, pos, nodelist=["CS"], ax=ax1, node_size=250, node_shape="s", node_color="#1a9850")
    ax1.set_title(f"Dynamic WSN Snapshot (t={time_window-1})", fontsize=11, fontweight="bold")
    ax1.set_xlim(0, AREA_WIDTH)
    ax1.set_ylim(0, AREA_HEIGHT)
    ax1.grid(True, linestyle="--", alpha=0.3)

    # 2. Top middle: Normalized Adjacency Heatmap (Spatial)
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(adj_tensor[-1].numpy(), cmap="viridis", interpolation="nearest")
    ax2.set_title("Normalized Adjacency Matrix $\\tilde{A}$", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Node Index")
    ax2.set_ylabel("Node Index")
    fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.04)

    # 3. Top right: Temporal Connectivity Dynamics
    ax3 = fig.add_subplot(gs[0, 2])
    timesteps = [s[0] for s in connectivity_stats]
    edges = [s[2] for s in connectivity_stats]
    comps = [s[3] for s in connectivity_stats]
    ax3.plot(timesteps, edges, marker="o", color="#2b83ba", label="Active Links (Edges)", linewidth=2)
    ax3.plot(timesteps, comps, marker="s", color="#d7191c", label="Connected Components", linewidth=2)
    ax3.set_title("Temporal Connectivity Dynamics", fontsize=11, fontweight="bold")
    ax3.set_xlabel("Timestep $t$")
    ax3.set_ylabel("Count")
    ax3.legend(loc="best")
    ax3.grid(True, linestyle="--", alpha=0.4)

    # 4. Bottom left: Node Embedding Heatmap (Spatial-Temporal Representation)
    ax4 = fig.add_subplot(gs[1, 0:2])
    node_labels = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]
    im4 = ax4.imshow(node_emb.numpy(), aspect="auto", cmap="magma", interpolation="nearest")
    ax4.set_yticks(range(len(node_labels)))
    ax4.set_yticklabels(node_labels, fontsize=8)
    ax4.set_title("ST-GNN Latent Node Embeddings $Z_{nodes} \\in \\mathbb{R}^{22 \\times 64}$", fontsize=11, fontweight="bold")
    ax4.set_xlabel("Latent Feature Dimensions (0 - 63)")
    fig.colorbar(im4, ax=ax4, fraction=0.02, pad=0.02)

    # 5. Bottom right: Global Graph Embedding & Model Summary
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.bar(range(len(graph_emb[:16])), graph_emb[:16].numpy(), color="#5e3c99", alpha=0.85)
    ax5.set_title("Global Graph Embedding (First 16 Dims)", fontsize=11, fontweight="bold")
    ax5.set_xlabel("Embedding Dim")
    ax5.set_ylabel("Activation Value")
    ax5.grid(True, linestyle="--", alpha=0.3)

    summary_text = (
        "ST-GNN Specifications:\n"
        "-----------------------\n"
        f"Input Node Features:  8 (pos, vel, energy, prio, type)\n"
        f"Spatial GCN Layers:   2 (GCN 8->32->32)\n"
        f"Temporal Model:       GRU (32->64, T={time_window})\n"
        f"Output Dimensions:    Node: (22, 64), Graph: (64)\n"
        f"Status:               Validated & Functional"
    )
    ax5.text(
        0.05, 0.95, summary_text,
        transform=ax5.transAxes,
        verticalalignment="top",
        fontsize=8.5,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="#f0f0f0", alpha=0.9)
    )

    plt.suptitle("Step 5: Spatio-Temporal Graph Neural Network (ST-GNN) Output", fontsize=15, fontweight="bold")
    plt.tight_layout()
    output_filename = "step5_st_gnn_output.png"
    plt.savefig(output_filename, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved Step 5 visualization to {output_filename}")
    print("Step 5 completed successfully!")


if __name__ == "__main__":
    run_step5()
