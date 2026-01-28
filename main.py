import os
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from multicore import multicore
from BFS_placer import bfs_placer
from Random_placer import random_placer
from Qubo_placer import qubo_placer
from TRHQA_placer import trhqa_placer
from Graph_placer import graph_placer 
from GTRHQA_placer import gtrhqa_placer
from CongestionHqa_placer import chqa_placer
from HQA_placer import hqa_placer

def run_simulation(cfg=None):
    """
    runs simulation given the configuration parameters
    
    """
    from pytket import qasm
    circuit_folders = cfg.get('circuit_folders')
    repeats = int(cfg.get('repeats'))
    out_csv = cfg.get('out_csv')
    plot_file = cfg.get('plot_file')
    multicore_conf = {
        'num_cores_x': int(cfg.get('num_cores_x', 8)),
        'num_cores_y': int(cfg.get('num_cores_y', 8)),
        'ncm_qubits': int(cfg.get('ncm_qubits', 4)),
        'ncp_qubits': int(cfg.get('ncp_qubits', 2)),
        'edge_weight_range': tuple(cfg.get('edge_weight_range', (1.0, 3.0))),
        'network_type': cfg.get('topology', 'all_to_all')
    }
    placers_to_run = [p for p in cfg.get('placers')]

    placers_instances = {}
    if 'TRHQA' in placers_to_run:
        placers_instances['TRHQA'] = trhqa_placer()
    if 'QUBO' in placers_to_run:
        placers_instances['QUBO'] = qubo_placer()
    if 'HQA' in placers_to_run:
        placers_instances['HQA'] = hqa_placer()
    if 'GTRHQA' in placers_to_run:
        placers_instances['GTRHQA'] = gtrhqa_placer()
    if 'RANDOM' in placers_to_run:
        placers_instances['RANDOM'] = random_placer()
    if 'BFS' in placers_to_run:
        placers_instances['BFS'] = bfs_placer()

    try:
        multicore_arch = multicore(
            multicore_conf['num_cores_x'],
            multicore_conf['num_cores_y'],
            ncm_qubits=multicore_conf['ncm_qubits'],
            ncp_qubits=multicore_conf['ncp_qubits'],
            edge_weight_range=multicore_conf['edge_weight_range'],
            network_type=multicore_conf.get('network_type')
        )
    except Exception as e:
        print("Warning: multicore init failed with args, falling back to default ctor:", e)
        multicore_arch = multicore(4, 4)

    qasm_list = []
    for folder in circuit_folders:
        if not os.path.exists(folder):
            print(f"Warning: circuit folder not found: {folder}")
            continue
        for fn in sorted(os.listdir(folder)):
            if fn.lower().endswith('.qasm'):
                qasm_list.append(os.path.join(folder, fn))
    if not qasm_list:
        raise RuntimeError("No .qasm files found in circuit_folders.")

    print("Running simulations on the following QASM files:")
    print(qasm_list)

    results = []
    for qfn in qasm_list:
        print(f"\n=== Processing {qfn} ===")
        circuit = qasm.circuit_from_qasm(qfn, maxwidth=1024)
        file_name = os.path.basename(qfn)

        entry = {'file': qfn}

        def run_placer_n_times(placer_obj, placer_name, run_fn_name='place_per_timeslice', cost_fn_name='per_timeslice_cost'):
            times = []
            costs = []
            for _ in range(repeats):
                try:
                    t0 = time.perf_counter()
                    place_fn = getattr(placer_obj, run_fn_name)
                    part = place_fn(circuit, multicore_arch)
                    t1 = time.perf_counter()
                    times.append(t1 - t0)

                    cost_fn = getattr(placer_obj, cost_fn_name)
                    cost = cost_fn(circuit, part, multicore_arch)
                    costs.append(cost)
                except Exception as ex:
                    print(f"[{placer_name}] run failed: {ex}")
                    times.append(float('inf'))
                    costs.append(float('inf'))
            mean_time = float(np.mean([t for t in times if np.isfinite(t)]) if any(np.isfinite(t) for t in times) else float('inf'))
            mean_cost = float(np.mean([c for c in costs if np.isfinite(c)]) if any(np.isfinite(c) for c in costs) else float('inf'))
            return mean_time, mean_cost

        for placer in placers_to_run:
            if placer not in placers_instances:
                print(f"Placer '{placer}' not recognized, skipping.")
                continue
            pt, pc = run_placer_n_times(placers_instances[placer], placer, run_fn_name='place_per_timeslice')
            entry.update({f'{placer}_time_mean': pt, f'{placer}_cost_mean': pc})
            print(f"{placer}: Average Time={pt:.4f}s, Average Cost={pc:.4f}")

        results.append(entry)

    with open(out_csv, 'w', newline='') as f:
        cols = ['file']
        for placer in placers_to_run:
            cols.append(f'{placer}_time_mean')
            cols.append(f'{placer}_cost_mean')
            
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            for c in cols:
                if c not in r:
                    r[c] = float('inf')
            writer.writerow(r)
    print(f"\nWrote all results to {out_csv}")

    files = [r['file'] for r in results]
    order = np.argsort(files)
    files = [files[i] for i in order]
    x = np.arange(len(files))
    plt.figure(figsize=(14,6))
    plt.subplot(1,2,1)
    plt.title('Average Runtime in s')
    width = 0.12
    offsets = [-1.5, -0.5, 0.5, 1.5, 2.5]

    for placer in placers_to_run:
        placer_sorted = [results[i][f'{placer}_time_mean'] for i in order]
        plt.bar(x + offsets[placers_to_run.index(placer)] * width, placer_sorted, width=width, label=placer)
    plt.xticks(x, files, rotation=45, ha='right')
    plt.xlabel("#gates")
    plt.ylabel("#time")
    plt.legend()

    plt.subplot(1,2,2)
    plt.title('Average Cost')
    for placer in placers_to_run:
        placer_sorted = [results[i][f'{placer}_cost_mean'] for i in order]
        plt.bar(x + offsets[placers_to_run.index(placer)] * width, placer_sorted, width=width, label=placer)
    plt.xticks(x, files, rotation=45, ha='right')
    plt.xlabel("#gates")
    plt.ylabel("#expected intercore communication")
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_file}")
    print("\nDone. Results summary:")
    for r in results:
        print(r)


if __name__ == '__main__':
    import argparse
    from config_parser import parse_config_file

    parser = argparse.ArgumentParser(description='Main entry for quantumsim. If --config is provided, run with that config.')
    parser.add_argument('--config', help='Path to text config file (see example_config.txt)')
    args = parser.parse_args()

    if args.config:
        try:
            cfg = parse_config_file(args.config)
        except AssertionError as e:
            print(f"Configuration error: {e}")
            raise

        print('Parsed config:')
        for k, v in cfg.items():
            print(f'  {k} = {v}')

        mode = cfg.get('mode')
        if mode == 'simulate':
            print('Running standard simulation')
            run_simulation(cfg)
        else:
            raise RuntimeError(f"Unknown mode in config: {mode}")
    else:
        print("No config file provided. Please run with --config <path_to_config_file>")

