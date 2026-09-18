import os
import json
import csv
import numpy as np

def inspect():
    print("=" * 75)
    print("UAV-WSN SIMULATION DATASET: INSPECTION & SUMMARY")
    print("=" * 75)

    meta_file = os.path.join("dataset", "dataset_metadata.json")
    if os.path.exists(meta_file):
        with open(meta_file, "r") as f:
            meta = json.load(f)
        print(f"Dataset Title:   {meta['dataset_name']}")
        print(f"Institution:     {meta['institution']}")
        print(f"Total Episodes:  {meta['total_episodes']} (Steps/Ep: {meta['steps_per_episode']})")
        print(f"Graph Snapshots: {meta['total_graph_snapshots']}")
        print(f"Splits:")
        for s, info in meta["splits"].items():
            print(f"  • {s.upper():<6}: {info['episodes']} episodes ({info['snapshots']} snapshots, {info['ratio']*100:.0f}%)")
    print("-" * 75)

    # 1. Sensor Telemetry
    sensor_path = os.path.join("dataset", "sensor_telemetry.csv")
    if os.path.exists(sensor_path):
        with open(sensor_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        print(f"\n[1] Sensor Telemetry: {len(rows):,} rows | {len(reader.fieldnames)} columns")
        print(f"    Columns: {reader.fieldnames}")
        print("\n    Sample Rows (First 4 records):")
        header = f"{'Ep':<4} {'Step':<5} {'Split':<6} {'Node':<5} {'X':<8} {'Y':<8} {'Speed':<6} {'Energy':<7} {'Priority':<9} {'CanReachCS'}"
        print("    " + header)
        print("    " + "-" * len(header))
        for r in rows[:4]:
            print(f"    {r['episode_id']:<4} {r['timestep']:<5} {r['split']:<6} {r['node_id']:<5} {r['pos_x']:<8} {r['pos_y']:<8} {r['speed']:<6} {r['energy_remaining']:<7} {r['mission_priority']:<9} {r['can_reach_cs']}")

        # Priority tally
        p_counts = {}
        for r in rows:
            p = r["mission_priority"]
            p_counts[p] = p_counts.get(p, 0) + 1
        print(f"\n    Priority Distribution: P1: {p_counts.get('1', 0):,} | P2: {p_counts.get('2', 0):,} | P3 (Critical): {p_counts.get('3', 0):,}")

    # 2. UAV Telemetry
    uav_path = os.path.join("dataset", "uav_telemetry.csv")
    if os.path.exists(uav_path):
        with open(uav_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            uav_rows = list(reader)
        print(f"\n[2] UAV Telemetry: {len(uav_rows):,} rows | {len(reader.fieldnames)} columns")
        print(f"    Columns: {reader.fieldnames}")
        print("\n    Sample UAV Trajectory Records:")
        header_uav = f"{'Ep':<4} {'Step':<5} {'Split':<6} {'UAV_X':<8} {'UAV_Y':<8} {'Energy':<8} {'DistToCS':<10} {'CS_Connected':<13} {'CoveredSensors'}"
        print("    " + header_uav)
        print("    " + "-" * len(header_uav))
        for r in uav_rows[:4]:
            print(f"    {r['episode_id']:<4} {r['timestep']:<5} {r['split']:<6} {r['pos_x']:<8} {r['pos_y']:<8} {r['energy_remaining']:<8} {r['dist_to_cs']:<10} {r['cs_connected']:<13} {r['covered_sensors_count']}")

    # 3. Network Links
    links_path = os.path.join("dataset", "network_links.csv")
    if os.path.exists(links_path):
        with open(links_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            link_rows = list(reader)
        print(f"\n[3] Network Links: {len(link_rows):,} active link records")
        dists = [float(r['distance_m']) for r in link_rows]
        lqs = [float(r['link_quality']) for r in link_rows]
        print(f"    Link Distance: Avg={np.mean(dists):.2f}m (Min={np.min(dists):.2f}m, Max={np.max(dists):.2f}m)")
        print(f"    Link Quality:  Avg={np.mean(lqs):.3f} (Min={np.min(lqs):.3f}, Max={np.max(lqs):.3f})")

    # 4. NPZ Graph Tensors
    npz_path = os.path.join("dataset", "graph_snapshots.npz")
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        print(f"\n[4] Graph Tensors (NPZ for ST-GNN):")
        for k in data.files:
            print(f"    • {k:<20}: shape={data[k].shape}, dtype={data[k].dtype}")

    print("\n" + "=" * 75)
    print("READY FOR PROJECT REVIEW PRESENTATION")
    print("=" * 75)

if __name__ == "__main__":
    inspect()
