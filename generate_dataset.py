import os
import json
import csv
import numpy as np
import networkx as nx

from config import (
    AREA_WIDTH,
    AREA_HEIGHT,
    NUM_SENSORS,
    MAX_PRIORITY,
    UAV_COMMUNICATION_RANGE,
    SENSOR_COMMUNICATION_RANGE,
    COMMAND_STATION_POSITION,
    UAV_INITIAL_POSITION,
    DISRUPTION_X_MIN,
    DISRUPTION_X_MAX,
    DISRUPTION_Y_MIN,
    DISRUPTION_Y_MAX,
    ENABLE_DISRUPTION
)
from environment import SimulationEnvironment
from network import DynamicWSNGraph


def is_in_disruption(x, y):
    """Checks if a coordinate is within the RF disruption zone."""
    if not ENABLE_DISRUPTION:
        return False
    return (DISRUPTION_X_MIN <= x <= DISRUPTION_X_MAX and
            DISRUPTION_Y_MIN <= y <= DISRUPTION_Y_MAX)


def generate_uav_wsn_dataset(
    output_dir="dataset",
    num_episodes=50,
    steps_per_episode=25,
    train_ratio=0.70,
    val_ratio=0.15,
    base_seed=1000
):
    """
    Generates a structured, multi-modal UAV-WSN dataset matching the exact schema:
      - Sensor Features: x, y, vx, vy, speed, energy, priority, in_disruption, degree, dist_to_cs, dist_to_uav
      - UAV Features: x, y, vx, vy, energy, dist_to_cs, cs_connected, avg_sensor_dist, covered_sensors_count
      - Network Features: source, target, distance, link_quality, is_active, in_disruption_link
      - Temporal Graph Snapshots: node feature matrices (N x F) and adjacency matrices (N x N)
      - Train / Val / Test split assignment
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 80)
    print(f"Generating UAV-WSN Simulation Dataset in: {output_dir}")
    print(f"Total Episodes: {num_episodes} | Steps/Episode: {steps_per_episode} | Total Snapshots: {num_episodes * steps_per_episode}")
    print("=" * 80)

    # Output file paths
    sensor_csv_path = os.path.join(output_dir, "sensor_telemetry.csv")
    uav_csv_path = os.path.join(output_dir, "uav_telemetry.csv")
    network_csv_path = os.path.join(output_dir, "network_links.csv")
    npz_path = os.path.join(output_dir, "graph_snapshots.npz")
    meta_path = os.path.join(output_dir, "dataset_metadata.json")

    # Determine splits per episode
    n_train = int(num_episodes * train_ratio)
    n_val = int(num_episodes * val_ratio)
    n_test = num_episodes - n_train - n_val

    def get_split(ep_idx):
        if ep_idx < n_train:
            return "train"
        elif ep_idx < n_train + n_val:
            return "val"
        else:
            return "test"

    # Headers
    sensor_fieldnames = [
        "episode_id", "timestep", "split", "node_id",
        "pos_x", "pos_y", "vel_x", "vel_y", "speed",
        "energy_remaining", "mission_priority", "in_disruption_zone",
        "node_degree", "dist_to_cs", "dist_to_uav", "can_reach_cs"
    ]

    uav_fieldnames = [
        "episode_id", "timestep", "split",
        "pos_x", "pos_y", "vel_x", "vel_y", "speed",
        "energy_remaining", "dist_to_cs", "cs_connected",
        "avg_sensor_dist", "covered_sensors_count", "network_components"
    ]

    network_fieldnames = [
        "episode_id", "timestep", "split",
        "source_node", "target_node", "distance_m",
        "link_quality", "is_active", "crosses_disruption"
    ]

    # Sensor CSV writer
    sensor_file = open(sensor_csv_path, "w", newline="", encoding="utf-8")
    sensor_writer = csv.DictWriter(sensor_file, fieldnames=sensor_fieldnames)
    sensor_writer.writeheader()

    # UAV CSV writer
    uav_file = open(uav_csv_path, "w", newline="", encoding="utf-8")
    uav_writer = csv.DictWriter(uav_file, fieldnames=uav_fieldnames)
    uav_writer.writeheader()

    # Network CSV writer
    network_file = open(network_csv_path, "w", newline="", encoding="utf-8")
    network_writer = csv.DictWriter(network_file, fieldnames=network_fieldnames)
    network_writer.writeheader()

    # Arrays for NPZ format (ST-GNN tensors)
    all_node_features = []
    all_adj_matrices = []
    all_episode_ids = []
    all_timesteps = []
    all_splits = []

    total_sensor_records = 0
    total_link_records = 0

    cs_pos = np.array(COMMAND_STATION_POSITION, dtype=float)

    for ep in range(num_episodes):
        split = get_split(ep)
        seed = base_seed + ep
        np.random.seed(seed)

        # Initialize environment & network
        sim_env = SimulationEnvironment()
        network = DynamicWSNGraph()

        # UAV moves along corridor relay trajectory towards (500, 260)
        uav_curr_pos = np.array(UAV_INITIAL_POSITION, dtype=float)
        uav_target_pos = np.array([500.0, 260.0], dtype=float)
        uav_energy = 100.0

        for t in range(steps_per_episode):
            # Advance sensor mobility
            if t > 0:
                sim_env.step()
                # Move UAV smoothly towards target
                direction = uav_target_pos - uav_curr_pos
                dist_to_target = np.linalg.norm(direction)
                if dist_to_target > 5.0:
                    step_size = min(30.0, dist_to_target)
                    uav_vel = (direction / dist_to_target) * step_size
                    uav_curr_pos += uav_vel
                else:
                    uav_vel = np.array([0.0, 0.0])
                uav_energy = max(0.0, uav_energy - 0.5)
            else:
                uav_vel = np.array([0.0, 0.0])

            sim_env.uav.position = (uav_curr_pos[0], uav_curr_pos[1])
            sim_env.uav.energy = uav_energy

            # Build graph
            graph = network.build_graph(sim_env)
            num_components = nx.number_connected_components(graph)

            # UAV metrics
            dist_uav_cs = np.linalg.norm(uav_curr_pos - cs_pos)
            uav_cs_connected = int(dist_uav_cs <= UAV_COMMUNICATION_RANGE)

            sensor_dists = []
            covered_count = 0

            # --- Log Sensors ---
            for sensor in sim_env.sensors:
                s_pos = np.array(sensor.position, dtype=float)
                s_dist_cs = np.linalg.norm(s_pos - cs_pos)
                s_dist_uav = np.linalg.norm(s_pos - uav_curr_pos)
                sensor_dists.append(s_dist_uav)

                if s_dist_uav <= UAV_COMMUNICATION_RANGE:
                    covered_count += 1

                s_vx = sensor.velocity * np.cos(sensor.direction)
                s_vy = sensor.velocity * np.sin(sensor.direction)
                s_speed = float(sensor.velocity)

                s_name = f"S{sensor.node_id}"
                deg = graph.degree(s_name) if graph.has_node(s_name) else 0
                can_reach_cs = int(nx.has_path(graph, s_name, "CS")) if graph.has_node(s_name) and graph.has_node("CS") else 0

                sensor_writer.writerow({
                    "episode_id": ep,
                    "timestep": t,
                    "split": split,
                    "node_id": s_name,
                    "pos_x": round(s_pos[0], 2),
                    "pos_y": round(s_pos[1], 2),
                    "vel_x": round(s_vx, 2),
                    "vel_y": round(s_vy, 2),
                    "speed": round(s_speed, 2),
                    "energy_remaining": round(sensor.energy, 2),
                    "mission_priority": sensor.priority,
                    "in_disruption_zone": int(is_in_disruption(s_pos[0], s_pos[1])),
                    "node_degree": deg,
                    "dist_to_cs": round(s_dist_cs, 2),
                    "dist_to_uav": round(s_dist_uav, 2),
                    "can_reach_cs": can_reach_cs
                })
                total_sensor_records += 1

            # --- Log UAV ---
            avg_s_dist = np.mean(sensor_dists) if sensor_dists else 0.0
            uav_writer.writerow({
                "episode_id": ep,
                "timestep": t,
                "split": split,
                "pos_x": round(uav_curr_pos[0], 2),
                "pos_y": round(uav_curr_pos[1], 2),
                "vel_x": round(uav_vel[0], 2),
                "vel_y": round(uav_vel[1], 2),
                "speed": round(np.linalg.norm(uav_vel), 2),
                "energy_remaining": round(uav_energy, 2),
                "dist_to_cs": round(dist_uav_cs, 2),
                "cs_connected": uav_cs_connected,
                "avg_sensor_dist": round(avg_s_dist, 2),
                "covered_sensors_count": covered_count,
                "network_components": num_components
            })

            # --- Log Network Links ---
            for u, v, d in graph.edges(data=True):
                p_u = graph.nodes[u]["position"]
                p_v = graph.nodes[v]["position"]
                dist = np.linalg.norm(np.array(p_u) - np.array(p_v))
                lq = d.get("link_quality", 0.5)
                crosses = int(is_in_disruption(p_u[0], p_u[1]) or is_in_disruption(p_v[0], p_v[1]))

                network_writer.writerow({
                    "episode_id": ep,
                    "timestep": t,
                    "split": split,
                    "source_node": u,
                    "target_node": v,
                    "distance_m": round(dist, 2),
                    "link_quality": round(lq, 4),
                    "is_active": 1,
                    "crosses_disruption": crosses
                })
                total_link_records += 1

            # --- ST-GNN Tensor Representation ---
            # Node order: S0..S19, UAV, CS (total 22 nodes)
            node_keys = [f"S{i}" for i in range(NUM_SENSORS)] + ["UAV", "CS"]
            N = len(node_keys)
            feat_dim = 6  # norm_x, norm_y, norm_vx, norm_vy, norm_energy, priority
            feats = np.zeros((N, feat_dim), dtype=np.float32)
            adj = np.zeros((N, N), dtype=np.float32)

            for idx, k in enumerate(node_keys):
                if k.startswith("S"):
                    s_idx = int(k[1:])
                    s = sim_env.sensors[s_idx]
                    sv_x = s.velocity * np.cos(s.direction)
                    sv_y = s.velocity * np.sin(s.direction)
                    feats[idx, 0] = s.position[0] / AREA_WIDTH
                    feats[idx, 1] = s.position[1] / AREA_HEIGHT
                    feats[idx, 2] = sv_x / 5.0
                    feats[idx, 3] = sv_y / 5.0
                    feats[idx, 4] = s.energy / 100.0
                    feats[idx, 5] = s.priority / MAX_PRIORITY
                elif k == "UAV":
                    feats[idx, 0] = uav_curr_pos[0] / AREA_WIDTH
                    feats[idx, 1] = uav_curr_pos[1] / AREA_HEIGHT
                    feats[idx, 2] = uav_vel[0] / 30.0
                    feats[idx, 3] = uav_vel[1] / 30.0
                    feats[idx, 4] = uav_energy / 100.0
                    feats[idx, 5] = 1.0
                elif k == "CS":
                    feats[idx, 0] = cs_pos[0] / AREA_WIDTH
                    feats[idx, 1] = cs_pos[1] / AREA_HEIGHT
                    feats[idx, 2] = 0.0
                    feats[idx, 3] = 0.0
                    feats[idx, 4] = 1.0
                    feats[idx, 5] = 1.0

            for idx1, k1 in enumerate(node_keys):
                for idx2, k2 in enumerate(node_keys):
                    if graph.has_edge(k1, k2):
                        adj[idx1, idx2] = graph[k1][k2].get("link_quality", 0.5)

            all_node_features.append(feats)
            all_adj_matrices.append(adj)
            all_episode_ids.append(ep)
            all_timesteps.append(t)
            all_splits.append(split)

        if (ep + 1) % 10 == 0 or ep == num_episodes - 1:
            print(f"  Processed Episode {ep+1:2d}/{num_episodes} ({split.upper()}) | Sensors: {total_sensor_records:,} rows | Links: {total_link_records:,} rows")

    sensor_file.close()
    uav_file.close()
    network_file.close()

    # Save NPZ bundle
    print("\nCompiling graph tensors into compressed NPZ archive...")
    np.savez_compressed(
        npz_path,
        node_features=np.array(all_node_features, dtype=np.float32),
        adjacency_matrices=np.array(all_adj_matrices, dtype=np.float32),
        episode_ids=np.array(all_episode_ids, dtype=np.int32),
        timesteps=np.array(all_timesteps, dtype=np.int32),
        splits=np.array(all_splits)
    )

    # Metadata JSON
    metadata = {
        "dataset_name": "UAV-WSN Mountain-Pass Surveillance Corridor Dataset",
        "description": "Simulation-generated spatio-temporal dataset capturing mobile sensor telemetry, UAV relay dynamics, and dynamic wireless link qualities under terrain-dependent RF disruptions.",
        "institution": "Kumaraguru College of Technology / KSI",
        "total_episodes": num_episodes,
        "steps_per_episode": steps_per_episode,
        "total_graph_snapshots": num_episodes * steps_per_episode,
        "num_sensors": NUM_SENSORS,
        "nodes_per_snapshot": NUM_SENSORS + 2,  # 20 sensors + UAV + CS = 22
        "splits": {
            "train": {"episodes": n_train, "snapshots": n_train * steps_per_episode, "ratio": train_ratio},
            "val": {"episodes": n_val, "snapshots": n_val * steps_per_episode, "ratio": val_ratio},
            "test": {"episodes": n_test, "snapshots": n_test * steps_per_episode, "ratio": 1.0 - train_ratio - val_ratio}
        },
        "files": {
            "sensor_telemetry_csv": {
                "filename": "sensor_telemetry.csv",
                "rows": total_sensor_records,
                "description": "Time-series observations of sensor coordinates, velocity, energy, priority, disruption flags, and CS connectivity."
            },
            "uav_telemetry_csv": {
                "filename": "uav_telemetry.csv",
                "rows": num_episodes * steps_per_episode,
                "description": "Time-series flight coordinates, battery state, distance to CS, and coverage metrics."
            },
            "network_links_csv": {
                "filename": "network_links.csv",
                "rows": total_link_records,
                "description": "Dynamic edge list specifying Euclidean distances, link qualities, and disruption crossing status."
            },
            "graph_snapshots_npz": {
                "filename": "graph_snapshots.npz",
                "arrays": ["node_features (T x 22 x 6)", "adjacency_matrices (T x 22 x 22)", "episode_ids", "timesteps", "splits"],
                "description": "Compressed binary tensors for Spatio-Temporal Graph Neural Network (ST-GNN) training and evaluation."
            }
        },
        "feature_schema": {
            "sensor_features": ["X, Y position (m)", "Velocity (vx, vy in m/s)", "Energy level (J)", "Mission priority (1, 2, 3)", "In disruption zone flag", "Node degree", "Distance to CS", "Distance to UAV"],
            "uav_features": ["UAV position (x, y in m)", "UAV energy (%)", "Sensor distance (avg, min)", "Command-station connectivity flag", "Active coverage count"],
            "network_features": ["Node connectivity (adjacency)", "Euclidean distance (m)", "Link quality (0.1 to 1.0)", "Dynamic topology (connected components)"],
            "temporal_features": ["T0 -> T1 -> ... -> Tn timesteps with graph evolution per step"]
        }
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "=" * 80)
    print("DATASET GENERATION COMPLETED SUCCESSFULLY!")
    print(f"1. Sensor Telemetry CSV:  {sensor_csv_path} ({total_sensor_records:,} rows)")
    print(f"2. UAV Telemetry CSV:     {uav_csv_path} ({num_episodes * steps_per_episode:,} rows)")
    print(f"3. Network Links CSV:     {network_csv_path} ({total_link_records:,} rows)")
    print(f"4. Graph Snapshots NPZ:   {npz_path} ({num_episodes * steps_per_episode:,} graph tensors)")
    print(f"5. Dataset Metadata JSON: {meta_path}")
    print("=" * 80)


if __name__ == "__main__":
    generate_uav_wsn_dataset()
