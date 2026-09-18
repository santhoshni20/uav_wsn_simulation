# UAV-WSN Mountain-Pass Surveillance Corridor Dataset

## 1. Overview
This dataset contains simulation-generated spatio-temporal telemetry for mobile wireless sensor nodes and an autonomous UAV aerial relay deployed in a rugged mountain-pass surveillance corridor under severe RF disruption.

The dataset matches the research requirements specified in the project review curriculum:
- **Sensor Features**: Positions ($X, Y$), velocity vectors ($V_x, V_y$), speed, energy levels, and mission priorities ($P_1, P_2, P_3$).
- **UAV Features**: Flight positions ($X, Y$), velocity, energy consumption, distance to base station, coverage count, and Command Station connectivity.
- **Network Features**: Active dynamic topology, node degree, Euclidean link distances, and link quality metrics ($R_{ij} \in [0.1, 1.0]$).
- **Temporal Snapshots**: $T_0 \to T_1 \to \dots \to T_n$ graph evolution across episodes.
- **Data Splits**: 70% Training (35 episodes), 15% Validation (7 episodes), 15% Testing (8 episodes).

---

## 2. Dataset Files Summary

| File Name | Format | Records | Description |
| :--- | :---: | :---: | :--- |
| `sensor_telemetry.csv` | CSV | 25,000 | Individual sensor mobility, battery, priority, and connectivity state at each timestep. |
| `uav_telemetry.csv` | CSV | 1,250 | UAV repositioning flight coordinates, remaining battery, and relay metrics per timestep. |
| `network_links.csv` | CSV | 38,347 | Detailed edge list of all active communication links with Euclidean distance and link quality. |
| `graph_snapshots.npz` | NPZ (NumPy) | 1,250 | Multi-dimensional tensors for Spatio-Temporal Graph Neural Network (ST-GNN) input. |
| `dataset_metadata.json` | JSON | — | Comprehensive schema definitions, parameters, and split boundaries. |

---

## 3. Data Dictionary & Feature Definitions

### (A) Sensor Telemetry (`sensor_telemetry.csv`)
| Column Name | Data Type | Units / Range | Description |
| :--- | :---: | :---: | :--- |
| `episode_id` | Integer | $0 - 49$ | Episode index representing an independent surveillance run. |
| `timestep` | Integer | $0 - 24$ | Discrete simulation time step ($T$). |
| `split` | String | `train`, `val`, `test` | Dataset partitioning label. |
| `node_id` | String | `S0` - `S19` | Unique identifier of the sensor node (20 mobile sensors). |
| `pos_x` | Float | $0.0 - 1000.0$ m | X coordinate within the simulation area. |
| `pos_y` | Float | $0.0 - 1000.0$ m | Y coordinate within the simulation area. |
| `vel_x` | Float | m/s | Velocity component along the X axis. |
| `vel_y` | Float | m/s | Velocity component along the Y axis. |
| `speed` | Float | $1.0 - 5.0$ m/s | Instantaneous scalar mobility speed. |
| `energy_remaining` | Float | $0.0 - 100.0$ J | Remaining battery energy of the sensor. |
| `mission_priority` | Integer | $1, 2, 3$ | $1 = \text{Low}$, $2 = \text{Elevated}$, $3 = \text{Mission-Critical}$. |
| `in_disruption_zone` | Integer | $0$ or $1$ | 1 if node is located within the RF shadowing dead zone. |
| `node_degree` | Integer | $\ge 0$ | Number of active 1-hop communication links. |
| `dist_to_cs` | Float | meters | Euclidean distance to Command Station at $(500, 100)$. |
| `dist_to_uav` | Float | meters | Euclidean distance to the UAV relay. |
| `can_reach_cs` | Integer | $0$ or $1$ | 1 if an active single-hop or multi-hop path exists to the CS. |

### (B) UAV Telemetry (`uav_telemetry.csv`)
| Column Name | Data Type | Units / Range | Description |
| :--- | :---: | :---: | :--- |
| `episode_id` | Integer | $0 - 49$ | Episode index. |
| `timestep` | Integer | $0 - 24$ | Discrete simulation time step ($T$). |
| `split` | String | `train`, `val`, `test` | Dataset split assignment. |
| `pos_x`, `pos_y` | Float | meters | Current UAV coordinates in the corridor ($1000 \times 1000$ m). |
| `vel_x`, `vel_y` | Float | m/s | UAV flight displacement vector. |
| `energy_remaining` | Float | $0.0 - 100.0$ % | Remaining UAV battery level. |
| `dist_to_cs` | Float | meters | Distance to the Command Station. |
| `cs_connected` | Integer | $0$ or $1$ | 1 if within UAV communication radius ($R = 300$ m) of CS. |
| `avg_sensor_dist` | Float | meters | Average distance to all deployed sensor nodes. |
| `covered_sensors_count` | Integer | $0 - 20$ | Number of sensors currently within UAV communication range. |
| `network_components` | Integer | $\ge 1$ | Number of partitioned subgraphs in the network topology. |

### (C) Network Links (`network_links.csv`)
| Column Name | Data Type | Description |
| :--- | :---: | :--- |
| `source_node` | String | Origin node (`S0` - `S19`, `UAV`, `CS`). |
| `target_node` | String | Destination node. |
| `distance_m` | Float | Euclidean distance between nodes in meters ($d \le 200$ m for sensors, $300$ m for UAV). |
| `link_quality` | Float | Link quality $R_{ij} \in [0.1, 1.0]$, degraded by 75% inside disruption zone. |
| `is_active` | Integer | 1 for active established wireless links ($R_{ij} \ge 0.3$). |
| `crosses_disruption` | Integer | 1 if link traverses the mountain-pass RF shadowing zone. |

### (D) Graph Tensors (`graph_snapshots.npz`)
- `node_features`: Tensor of shape `(1250, 22, 6)` encoding normalized coordinates, velocity, battery, and priority for all 22 network nodes (20 sensors + UAV + CS).
- `adjacency_matrices`: Tensor of shape `(1250, 22, 22)` encoding the dynamic weighted adjacency matrix with link quality values.
- `splits`: Array of length `1250` tagging each snapshot as `train`, `val`, or `test`.

---

## 4. Why Simulation-Generated? (Defense Justification for Review Panel)
1. **Dynamic Temporal Evolution**: Physical UAV-WSN deployments in hazardous mountain corridors require continuous topological changes ($T_0 \to T_1 \to \dots \to T_n$) that cannot be captured by static benchmark datasets.
2. **Ground Truth Controllability**: Real-world radio packet traces lack precise ground-truth channel matrices, instantaneous energy depletion rates, and repeatable RF disruption zones.
3. **Rigorous Fair Comparison**: The exact same simulation environment and mobility trajectories are used to benchmark the proposed ST-GNN + MAPPO model against all baseline methods (`Static+Dijkstra`, `Random+Greedy`, `Heuristic Centroid`, `Vanilla MAPPO`).
