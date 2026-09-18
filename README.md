# UAV-Assisted Mobile Wireless Sensor Network (WSN) Simulation in Remote Mountain-Pass Surveillance Corridor

## Current Scope Delivered: **Phase 1 – Environment & Data Pipeline**

This repository contains the simulation environment and data pipeline for an autonomous UAV-assisted Wireless Sensor Network deployed in a harsh mountain-pass surveillance corridor subjected to terrain-induced radio frequency (RF) shadowing and communication disruptions.

---

## 1. Phase 1 Overview & Delivered Steps

As per academic review milestones, this phase focuses exclusively on the foundational **Environment & Data Pipeline** comprising Steps 1 through 4:

| Step | Title | Description | Implementation File |
| :--- | :--- | :--- | :--- |
| **Step 1** | **Simulation Environment** | Constructs a $1000 \times 1000$ m surveillance region with an active mountain corridor $X \in [300, 700]$, $Y \in [100, 900]$ m, static Command Station at $(500, 100)$, and an aerial UAV relay. | [`environment.py`](environment.py), [`config.py`](config.py) |
| **Step 2** | **Mobile Sensor Simulation** | Simulates 20 ground sensor nodes moving with velocities $1 - 5$ m/s, reflection at boundary edges, dynamic battery depletion, and mission priorities ($P_1, P_2, P_3$). | [`nodes.py`](nodes.py), [`environment.py`](environment.py) |
| **Step 3** | **Dynamic WSN Graph** | Employs NetworkX to build dynamic graph topologies at every timestep based on Euclidean transmission thresholds ($R_s = 200$ m, $R_u = 300$ m) and channel link qualities. | [`network.py`](network.py) |
| **Step 4** | **Communication Disruptions** | Models deep RF shadowing within $X \in [450, 550], Y \in [350, 650]$ m with a 75% link quality degradation factor, inducing realistic network partitioning. | [`disruptions.py`](disruptions.py) |

---

## 2. Mathematical Modeling & System Formulation

### A. Channel Link Quality & Range Condition
Two nodes $i$ and $j$ at positions $\mathbf{p}_i$ and $\mathbf{p}_j$ establish a communication link if their Euclidean distance $d_{ij} \le R_{\text{comm}}$ and link quality $Q_{ij} \ge \gamma_{\text{threshold}} = 0.3$:
$$d_{ij} = \|\mathbf{p}_i - \mathbf{p}_j\|_2$$
$$Q_{ij}^{(0)} = 1 - \frac{d_{ij}}{R_{\text{comm}}}$$

### B. Terrain RF Disruption Zone
When the midpoint of a link $\mathbf{m}_{ij} = \frac{\mathbf{p}_i + \mathbf{p}_j}{2}$ falls inside the mountain-pass shadowing corridor:
$$Q_{ij} = \begin{cases} 
Q_{ij}^{(0)} \times 0.25 & \text{if } \mathbf{m}_{ij} \in \mathcal{Z}_{\text{disruption}} \\
Q_{ij}^{(0)} & \text{otherwise}
\end{cases}$$

### C. Node Mobility & Energy Depletion
Sensor nodes update coordinates with boundary reflection:
$$\mathbf{p}_i(t+1) = \mathbf{p}_i(t) + v_i \cdot [\cos(\theta_i), \sin(\theta_i)]^T \cdot \Delta t$$
$$E_i(t+1) = \max(0, E_i(t) - \epsilon_{\text{move}})$$

---

## 3. How to Run Phase 1 Simulation

### Quick Start
Execute the main entry point to run all 4 steps and generate the complete Phase 1 visualization:

```bash
# Using the virtual environment
.\venv\Scripts\python.exe main.py
```
*(Alternatively: `python phase1_demo.py`)*

### Console Output
The script prints detailed step-by-step telemetry:
- **Step 1**: Initial deployment coordinates, node energy, and priority levels.
- **Step 2**: 50-timestep trajectory analysis, total displacement, and battery depletion.
- **Step 3**: Dynamic graph metrics (node count, active edges, number of connected components, average link quality).
- **Step 4**: Partition frequency and disruption impact statistics.

---

## 4. Phase 1 Graphical Simulation Output

Running `main.py` automatically generates a multi-panel visual report saved to **`phase1_complete_output.png`**:

1. **Panel 1 (Top Left)**: Spatial layout depicting the mountain corridor, disruption zone, Command Station, UAV, and 20 sensor trajectory paths color-coded by mission priority.
2. **Panel 2 (Top Right)**: Continuous energy depletion curves across all 50 timesteps.
3. **Panel 3 (Center)**: Four discrete graph topology snapshots ($T=0, 10, 25, 49$), showing real-time edge establishment and broken links.
4. **Panel 4 (Bottom Left)**: Temporal evolution of active communication links vs. disconnected subgraphs (components).
5. **Panel 5 (Bottom Right)**: Average link quality trajectory over time highlighting network partitioning events.

---

## 5. Project Roadmap (Next Phases)

- **Phase 1: Environment & Data Pipeline** *(Completed)*
  - Steps 1–4: Environment, mobile nodes, dynamic graph, terrain disruptions.
- **Phase 2: Algorithm Design & RL Framework** *(Next Milestone)*
  - Steps 5–8: Spatio-Temporal Graph Neural Network (ST-GNN), Gymnasium RL environment, Multi-Agent PPO (MAPPO), multi-objective reward shaping.
- **Phase 3: Training, Evaluation & Demonstration** *(Final Milestone)*
  - Steps 9–12: Cooperative policy training, baseline comparisons (Dijkstra, Greedy, Heuristic), benchmark metrics, live demonstration.
