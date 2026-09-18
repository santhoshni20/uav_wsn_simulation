"""
=============================================================================
UAV-Assisted Mobile WSN Simulation in Mountain-Pass Surveillance Corridor
=============================================================================
PHASE 1: Environment & Data Pipeline
  Step 1 - Simulation Environment
  Step 2 - Mobile Sensor Simulation
  Step 3 - Dynamic WSN Graph
  Step 4 - Communication Disruptions

Run:
  python main.py
=============================================================================
"""

import sys
import numpy as np

from phase1_demo import (
    run_step1,
    run_step2,
    run_step3,
    run_step4,
    generate_phase1_visualization,
    SIMULATION_STEPS,
    NUM_SENSORS
)
from environment import SimulationEnvironment


def main():
    # Check if user requested live simulation
    if "--live" in sys.argv or "--animate" in sys.argv:
        from live_simulation import run_live_simulation
        print()
        print("+" + "=" * 65 + "+")
        print("|   UAV-WSN Live Interactive Simulation (Phase 1)                 |")
        print("|   Controls: Spacebar = Pause/Resume | R = Restart | Q = Quit    |")
        print("+" + "=" * 65 + "+")
        print()
        run_live_simulation()
        return

    print()
    print("+" + "=" * 65 + "+")
    print("|   UAV-WSN Mountain-Pass Surveillance Simulation                 |")
    print("|   PHASE 1: Environment & Data Pipeline (Steps 1 - 4)           |")
    print("+" + "=" * 65 + "+")
    print()

    # Step 1: Create simulation environment
    env = run_step1()

    # Step 2: Simulate mobile sensors
    trajectories, energy_history, uav_positions = run_step2(env)

    # Step 3: Build dynamic WSN graph
    network, graph_snapshots, topology_stats, env = run_step3(env, trajectories)

    # Re-run environment from seed 42 to capture aligned trajectories
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

    # Step 4: Introduce and analyze communication disruptions
    run_step4(graph_snapshots, topology_stats)

    # Visualization
    print("=" * 65)
    print("  VISUALIZATION - Generating Phase 1 Graphical Analysis")
    print("=" * 65)
    output_file = generate_phase1_visualization(
        traj2, ehist2, graph_snapshots, topology_stats, env2
    )

    print()
    print("+" + "=" * 65 + "+")
    print("|   [SUCCESS] PHASE 1 EXECUTION COMPLETED                         |")
    print("|                                                                 |")
    print(f"|   Output Plot : {output_file:<46}  |")
    print("|                                                                 |")
    print("|   Milestones Delivered:                                         |")
    print("|     [x] Step 1: Terrain, corridor, sensor & UAV initialization  |")
    print("|     [x] Step 2: 20 mobile sensor random-walk & energy models    |")
    print("|     [x] Step 3: Dynamic NetworkX spatial graph generation       |")
    print("|     [x] Step 4: RF disruption zone & link degradation analysis  |")
    print("|                                                                 |")
    print("|   [TIP] To watch the real-time moving sensor simulation:        |")
    print("|         python main.py --live                                   |")
    print("+" + "=" * 65 + "+")
    print()


if __name__ == "__main__":
    main()