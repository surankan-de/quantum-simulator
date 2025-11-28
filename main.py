import networkx as nx
import random
from collections import deque
from pytket import Circuit, OpType,qasm
import copy
from BfsPlacer import bfs_placer
from RandomPlacer import random_placer
from HqaPlacer import hqa_placer
from AppHqa import apphqa_placer
import os
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

class cp_qubit:
    def __init__(self, id, qpu):
        self.qid = id
        self.qpu = qpu
        self.occupied = False
        self.job_id = None

    def allocate(self, job_id):
        self.occupied = True
        self.job_id = job_id


class cm_qubit:
    def __init__(self, id, qpu):
        self.qid = id
        self.qpu = qpu
        self.occupied = False
        self.job_id = None

    def allocate(self, job_id):
        self.occupied = True
        self.job_id = job_id


class qpu:
    def __init__(self, id, ncm_qubits, ncp_qubits):
        self.qpuid = id
        self.occupied = False
        self.job_id = []
        self.job_status = {}
        self.ncm_qubits = ncm_qubits
        self.ncp_qubits = ncp_qubits
        self.cm_qubits = []
        self.cp_qubits = []
        self.init_qpu()
        self.available_qubits = ncp_qubits
        self.pending_comm = 0
        self.collaboration_data = None
    def reset_pending_comm(self):
        """Reset the pending communication count (called after each timeslice)."""
        self.pending_comm = 0

    def allocate_qubits(self, job_id, n):
        self.occupied = True
        self.job_id.append(job_id)
        self.available_qubits -= n

    def init_qpu(self):
        for i in range(self.ncm_qubits):
            self.cm_qubits.append(cm_qubit(i, self))
        for i in range(self.ncp_qubits):
            self.cp_qubits.append(cp_qubit(i, self))

    def allocate_job(self, job, n_qubits):
        self.occupied = True
        self.job_id.append(job.id)
        self.job_status[job.id] = 'running'
        self.available_qubits -= n_qubits

    def free_qubits(self, n_qubits, job):
        self.job_status[job.id] = 'finished'
        self.available_qubits += n_qubits


class multicore:
    def __init__(self, num_qpus_x, num_qpus_y, ncm_qubits=4, ncp_qubits=16, edge_weight_range=(1, 10)):
        print(f"numx {num_qpus_x}, numy {num_qpus_y}")
        self.network = nx.grid_2d_graph(num_qpus_x, num_qpus_y)
        
        # Add random weights to all edges
        self._add_edge_weights(edge_weight_range)

        self.qpus = []
        self.collaboration_data = None
        
        # Correctly iterate through the grid graph nodes
        node_id = 0
        for node in self.network.nodes:
            qpu_instance = qpu(node_id, ncm_qubits, ncp_qubits)
            
            self.network.nodes[node]['type'] = 'qpu'
            self.network.nodes[node]['qpu'] = qpu_instance
            self.network.nodes[node]['qpu'].pending_comm = 0
            self.qpus.append(qpu_instance)
            self.network.nodes[node]['available_qubits'] = [qubit for qubit in qpu_instance.cp_qubits if
                                                            not qubit.occupied]
            node_id += 1

        print("Node ID: ", node_id)
        self.qpu_qubit_num = ncp_qubits
        self.ncm_qubits = ncm_qubits
        
        print(f"Network created with weighted edges. Weight range: {edge_weight_range}")
        print(f"Sample edge weights: {list(self.network.edges(data='weight'))[:5]}")

    def _add_edge_weights(self, weight_range):
        """
        Add random weights to all edges in the network.
        
        Args:
            weight_range: Tuple (min_weight, max_weight) for random weight generation
        """
        min_weight, max_weight = weight_range
        
        for edge in self.network.edges():
            # Generate random weight for each edge
            weight = random.uniform(min_weight, max_weight)
            self.network[edge[0]][edge[1]]['weight'] = weight
    
    def _add_custom_edge_weights(self, weight_function=None):
        """
        Add custom weights to edges based on a function.
        
        Args:
            weight_function: Function that takes (node1, node2) and returns weight
        """
        if weight_function is None:
            # Default: distance-based weights with some randomness
            def default_weight_func(node1, node2):
                # Base weight on Manhattan distance with random factor
                manhattan_dist = abs(node1[0] - node2[0]) + abs(node1[1] - node2[1])
                base_weight = manhattan_dist + 1  # Ensure minimum weight of 1
                random_factor = random.uniform(0.5, 2.0)  # Add 50% randomness
                return base_weight * random_factor
            
            weight_function = default_weight_func
        
        for edge in self.network.edges():
            weight = weight_function(edge[0], edge[1])
            self.network[edge[0]][edge[1]]['weight'] = weight

    def test_legal(self, partition):
        for qpu_id in partition.keys():
            if len(partition[qpu_id]) > self.network.nodes[qpu_id]['qpu'].available_qubits:
                return False
        return True

    def calculate_cost_qpu(self, partition, circuit):
        """
        Calculate cost using weighted shortest paths
        """
        cost = 0
        for command in circuit:
            type = command.op.type
            qubits = command.qubits
            if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                if partition[qubits[0]] != partition[qubits[1]]:
                    # Use weighted shortest path
                    try:
                        distance = nx.shortest_path_length(self.network, 
                                                         partition[qubits[0]], 
                                                         partition[qubits[1]], 
                                                         weight='weight')
                    except nx.NetworkXNoPath:
                        # Fallback to unweighted if no path found (shouldn't happen in grid)
                        distance = nx.shortest_path_length(self.network, 
                                                         partition[qubits[0]], 
                                                         partition[qubits[1]])
                    cost += distance
        return cost
        
    def place_circuit_randomly(self, circuit, placer):
        return placer.place(circuit, self)

    def place_circuit_bfs(self, circuit, placer):
        return placer.place(circuit, self)

    def get_edge_weights_summary(self):
        """
        Get summary statistics of edge weights
        """
        weights = [data['weight'] for _, _, data in self.network.edges(data=True)]
        return {
            'min': min(weights),
            'max': max(weights),
            'mean': np.mean(weights),
            'std': np.std(weights)
        }


def is_partition_legal(partition, multicore_arch, timeslices=None):
    """
    Check if a given timeslice partition is legal.
    partition: list of dicts (one per timeslice) mapping logical qubit -> QPU coordinate
    multicore_arch: multicore instance (contains QPU capacity info)
    timeslices: optional list of timeslices (if available) to check if a qubit is used in multiple gates

    Returns True if legal, False otherwise.
    """
    for timeslice_idx, timeslice_mapping in enumerate(partition):
        qpu_qubit_count = {}
        for qpu_coord in timeslice_mapping.values():
            qpu_qubit_count[qpu_coord] = qpu_qubit_count.get(qpu_coord, 0) + 1
        
        for qpu_coord, count in qpu_qubit_count.items():
            qpu_capacity = multicore_arch.qpu_qubit_num
            if count > qpu_capacity:
                print(f"[Illegal] Timeslice {timeslice_idx}, QPU {qpu_coord} has {count} qubits (capacity {qpu_capacity})")
                return False

        if len(timeslice_mapping) != len(set(timeslice_mapping.keys())):
            print(f"[Illegal] Timeslice {timeslice_idx} has duplicate logical qubit assignments.")
            return False

        if timeslices:
            used_qubits = set()
            for gate in timeslices[timeslice_idx]:
                if hasattr(gate, "qubits"):
                    involved_qubits = [q.index[0] for q in gate.qubits]  
                else:
                    involved_qubits = gate[1]
                for q in involved_qubits:
                    if q in used_qubits:
                        print(f"[Illegal] Timeslice {timeslice_idx}: Logical qubit {q} used in multiple gates.")
                        return False
                    used_qubits.add(q)

    return True


def run_simulation(multicore_arch, circuit_directory):
    """
    Simulates different placement algorithms for all QASM circuits in a directory.
    Now includes GraphPlacer with extensive analysis.
    """

    import time
    weight_stats = multicore_arch.get_edge_weights_summary()
    print(f"Edge weight statistics: {weight_stats}")

    # 2. Initialize placers
    random_placer_instance = random_placer()
    bfs_placer_instance = bfs_placer()
    hqa_placer_instance = hqa_placer()
    apphqa_placer_instance = apphqa_placer()

    # 3. Store results
    results = {}
    results["Random Placer"] = []
    results["BFS Placer"] = []
    results["HQA Placer"] = []
    results["apphqa Placer"] =[]

    
    circuit_names = []
    gates_counts = []

    # 4. Iterate through all files in the directory
    for filepath in circuit_directory:
        for filename in os.listdir(filepath):
            if filename.endswith(".qasm"):
                file_path = os.path.join(filepath, filename)

                # Load the circuit
                circuit = qasm.circuit_from_qasm(file_path, maxwidth = 1024)
                
                num_gates = circuit.n_gates
                gates_counts.append(num_gates)
                circuit_names.append(filename)

                # Calculate cost for each placer
                print(f"\nProcessing circuit: {filename} with {num_gates} gates")

                # Random Placer
                start_time = time.time()
                random_perslice_partition = random_placer_instance.place_per_timeslice_weighted_aware(circuit, multicore_arch)
                random_cost = random_placer_instance.per_timeslice_cost(circuit, random_perslice_partition, multicore_arch)
                end_time = time.time()
                random_placer_instance.distribution_matrix(circuit, random_perslice_partition, multicore_arch)
                results['Random Placer'].append(random_cost)
                print(f"Random Placer cost: {random_cost}, time: {end_time - start_time:.4f} seconds")
                is_partition_legal(random_perslice_partition, multicore_arch)
                print(f"Ended Random Placer")

                # BFS Placer
                start_time = time.time()
                bfs_perslice_partition = bfs_placer_instance.place_per_timeslice(circuit, multicore_arch)
                bfs_cost = bfs_placer_instance.per_timeslice_cost(circuit, bfs_perslice_partition, multicore_arch)
                end_time = time.time()
                results['BFS Placer'].append(bfs_cost)
                print(f"BFS Placer cost: {bfs_cost}, time: {end_time - start_time:.4f} seconds")
                is_partition_legal(bfs_perslice_partition, multicore_arch)
                print(f"Ended BFS Placer")

                
                # HQA Placer (now uses weighted costs)
                start_time = time.time()
                hqa_perslice_partition = hqa_placer_instance.place_per_timeslice(circuit, multicore_arch)
                hqa_cost = hqa_placer_instance.per_timeslice_cost(circuit, hqa_perslice_partition, multicore_arch)
                end_time = time.time()
                results['HQA Placer'].append(hqa_cost)
                print(f"HQA Placer cost: {hqa_cost}, time: {end_time - start_time:.4f} seconds")
                is_partition_legal(hqa_perslice_partition, multicore_arch)
                print(f"Ended HQA Placer")

                # apphqa Placer
                start_time = time.time()
                apphqa_perslice_partition = apphqa_placer_instance.place_per_timeslice(circuit, multicore_arch)
                apphqa_cost = apphqa_placer_instance.per_timeslice_cost(circuit, apphqa_perslice_partition, multicore_arch)
                end_time = time.time()
                results['apphqa Placer'].append(apphqa_cost)
                print(f"apphqa Placer cost: {apphqa_cost}, time: {end_time - start_time:.4f} seconds")
                is_partition_legal(apphqa_perslice_partition, multicore_arch)
                print(f"Ended apphqa Placer")

                print("-" * 50)


    return gates_counts, results


def print_comparative_summary(results):
    """
    Print a comparative summary of all placement algorithms.
    """
    print("\n" + "="*60)
    print("COMPARATIVE SUMMARY")
    print("="*60)
    
    for placer_name, costs in results.items():
        if costs:  # Only process if we have results
            avg_cost = np.mean(costs)
            min_cost = min(costs)
            max_cost = max(costs)
            std_cost = np.std(costs)
            
            print(f"\n{placer_name}:")
            print(f"  Average Cost: {avg_cost:.4f}")
            print(f"  Best Cost: {min_cost:.4f}")
            print(f"  Worst Cost: {max_cost:.4f}")
            print(f"  Std Dev: {std_cost:.4f}")
    
    print("\n" + "="*60)


                
def give_partition_with_fixed_hops(multicore_arch, circuit_file, t, max_iters=500):
    """
    Fast version: start from HQA partition and greedily increase total weighted-hop
    cost toward t.  Uses local delta cost updates (no deepcopy).
    """

    import os, random, networkx as nx
    from pytket import qasm
    from HqaPlacer import hqa_placer

    partitions = []

    # --- precompute distances once ---
    G = multicore_arch.network
    dist = dict(nx.all_pairs_dijkstra_path_length(G, weight="weight"))
    cores = list(G.nodes)
    qpu_cap = multicore_arch.qpu_qubit_num

    def get_dist(a, b):
        if a == b:
            return 0.0
        return dist.get(a, {}).get(b, float("inf"))

    def is_legal(mapping):
        counts = {}
        for q, c in mapping.items():
            counts[c] = counts.get(c, 0) + 1
        return all(cnt <= qpu_cap for cnt in counts.values())

    # --------------------------------------------------------------------
    for folder in circuit_file:
        for fn in os.listdir(folder):
            if not fn.endswith(".qasm"):
                continue
            path = os.path.join(folder, fn)
            circuit = qasm.circuit_from_qasm(path)
            print(circuit)
            placer = hqa_placer()
            timeslices = placer.break_into_timeslices(circuit)
            partition = placer.place_per_timeslice(circuit, multicore_arch)
            qubits = list(circuit.qubits)
            num_slices = len(timeslices)

            # --- helper to compute total cost ---
            def total_cost(part):
                cost = 0.0
                for s, ops in enumerate(timeslices):
                    m = part[s]
                    for g in ops:
                        if len(g.qubits) == 2:
                            q1, q2 = g.qubits
                            cost += get_dist(m[q1], m[q2])
                for s in range(1, num_slices):
                    prev, curr = part[s-1], part[s]
                    for q in qubits:
                        cost += get_dist(prev[q], curr[q])
                return cost

            curr_cost = total_cost(partition)
            print(f"[{fn}] HQA base cost={curr_cost:.3f}, target={t}")
            if curr_cost >= t:
                print(f"[{fn}] already ≥ target, using HQA partition")
                partitions.append(partition)
                continue

            # ---------------- main loop -----------------
            tol = 1e-3
            best_diff = abs(curr_cost - t)
            stagnation = 0
            for it in range(max_iters):
                improved = False
                # pick random slice/qubit to perturb (avoid scanning all)
                s = random.randrange(num_slices)
                q = random.choice(qubits)
                old_core = partition[s][q]
                # try a few random new cores farther away
                candidate_cores = sorted(cores, key=lambda c: get_dist(old_core, c), reverse=True)[:3]
                for new_core in candidate_cores:
                    if new_core == old_core:
                        continue
                    # capacity check
                    occ = sum(1 for v in partition[s].values() if v == new_core)
                    if occ >= qpu_cap:
                        continue
                    partition[s][q] = new_core
                    if not is_legal(partition[s]):
                        partition[s][q] = old_core
                        continue
                    new_cost = total_cost(partition)
                    if new_cost > curr_cost and abs(new_cost - t) < best_diff:
                        curr_cost = new_cost
                        best_diff = abs(new_cost - t)
                        improved = True
                        stagnation = 0
                        break
                    else:
                        # revert
                        partition[s][q] = old_core
                if not improved:
                    stagnation += 1
                if best_diff < tol or stagnation > 50:
                    break
                if it % 20 == 0:
                    print(f"  iter {it}: cost={curr_cost:.3f}, diff={best_diff:.3f}")

            print(f"[give_partition_with_fixed_hops] {fn} -> target={t}, achieved={curr_cost:.3f}")
            partitions.append(partition)
    return partitions


def get_routing_workload_from_partition(circuit_file, partition):
    """
    Calculates the inter-timeslice routing workload (qubit movement between cores)
    based on a pre-computed partition.

    Args:
        circuit_file (list[str]): List of folders containing the QASM files.
                                 (Used here only to load circuit info for timeslice structure).
        partition (list[dict]): A list where partition[s] is a dictionary 
                                mapping qubit to core for timeslice s.
                                e.g., [{'q[0]': 'C1', 'q[1]': 'C0'}, ...]

    Returns:
        dict: A dictionary where keys are circuit filenames and values are 
              the detailed routing workload across all timeslice transitions.
    """

    import os
    from pytket import qasm
    from HqaPlacer import hqa_placer # Assuming this is available to define timeslices
    from collections import defaultdict

    all_circuits_workloads = {}

    # NOTE: The provided 'partition' should correspond to a specific circuit.
    # Since the original function iterates over files, we'll try to match 
    # the partition structure based on the first circuit found. 
    # A more robust function would take the 'circuit' object directly as input.
    
    # --- Simplified structure to extract circuit details ---
    found_circuit_details = False
    for folder in circuit_file:
        for fn in os.listdir(folder):
            if not fn.endswith(".qasm"):
                continue
            
            path = os.path.join(folder, fn)
            circuit = qasm.circuit_from_qasm(path)
            
            # Need hqa_placer to determine the timeslice structure
            placer = hqa_placer()
            timeslices = placer.break_into_timeslices(circuit)
            
            qubits = list(circuit.qubits)
            num_slices = len(timeslices)
            found_circuit_details = True
            break
        if found_circuit_details:
            break
            
    if not found_circuit_details:
        print("Error: No QASM files found in the specified path.")
        return {}
        
    # --- Calculate inter-timeslice routing workload ---
    
    circuit_workload = {}
    
    # We use the length of the input partition to determine the number of transitions
    if len(partition) != num_slices:
         print("Warning: Partition length does not match circuit timeslice count.")
         num_slices = len(partition)

    for s in range(1, num_slices):
        prev_map = partition[s-1]
        curr_map = partition[s]
        
        # Dictionary to count movements from (src_core) to (dst_core)
        core_movement_counts = defaultdict(int)
        
        # We only iterate over the qubits present in the partition mappings
        qubits_in_map = set(prev_map.keys()) & set(curr_map.keys())
        
        for q in qubits_in_map:
            src_core = prev_map[q]
            dst_core = curr_map[q]
            
            # If the core assignment changes, it requires routing
            if src_core != dst_core:
                core_movement_counts[(src_core, dst_core)] += 1
                
        # Format the result for this timeslice transition (s-1 -> s)
        inter_slice_workload = []
        for (src, dst), count in core_movement_counts.items():
            # Format: [src_core, dst_core, num_qubits_to_send]
            inter_slice_workload.append([src, dst, count])
        
        # Store the workload for this transition
        circuit_workload[f'slice_{s-1}_to_{s}'] = inter_slice_workload


    all_circuits_workloads[fn] = circuit_workload
            
    return all_circuits_workloads




def generate_routing_heatmap_data_simple(workload_data, multicore_arch):
    G = multicore_arch.network
    cores = list(G.nodes)
    core_to_index = {core: i for i, core in enumerate(cores)}
    num_cores = len(cores)
    
    core_coordinates = {core: G.nodes[core].get('pos', (core[0], core[1])) for core in cores}

    transfer_matrix = np.zeros((num_cores, num_cores), dtype=int)

    # Build transfer matrix
    for filename, workload in workload_data.items():
        for slice_key, transfers in workload.items():
            for src, dst, count in transfers:
                if src in core_to_index and dst in core_to_index:
                    transfer_matrix[core_to_index[src], core_to_index[dst]] += count

    # Compute inbound / outbound / total
    inbound_traffic = np.sum(transfer_matrix, axis=0)
    outbound_traffic = np.sum(transfer_matrix, axis=1)
    total_traffic_per_core = inbound_traffic + outbound_traffic

    return (
        transfer_matrix,
        core_coordinates,
        cores,
        total_traffic_per_core,
        inbound_traffic,
        outbound_traffic,
    )


def plot_core_traffic_heatmap_simple(cores, total_traffic, core_coords_dict,
                                     inbound_traffic=None, outbound_traffic=None):
    if not total_traffic.any():
        print("No routing traffic to plot.")
        return

    x = [core_coords_dict[c][0] for c in cores]
    y = [core_coords_dict[c][1] for c in cores]
    x_dim, y_dim = max(x) + 1, max(y) + 1

    # Helper to fill a grid
    def fill_grid(values):
        grid = np.zeros((y_dim, x_dim))
        for i, core in enumerate(cores):
            cx, cy = core_coords_dict[core]
            if cx < x_dim and cy < y_dim:
                grid[cy, cx] = values[i]
        return grid

    total_grid = fill_grid(total_traffic)
    inbound_grid = fill_grid(inbound_traffic) if inbound_traffic is not None else None
    outbound_grid = fill_grid(outbound_traffic) if outbound_traffic is not None else None

    ncols = 3 if inbound_grid is not None and outbound_grid is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 5))

    if ncols == 1:
        axes = [axes]

    # Inbound
    if inbound_grid is not None:
        im1 = axes[0].imshow(inbound_grid, cmap='Blues', interpolation='nearest')
        axes[0].set_title('Inbound Qubit Transfers')
        axes[0].invert_yaxis()
        plt.colorbar(im1, ax=axes[0], label='Inbound Traffic')

    # Outbound
    if outbound_grid is not None:
        im2 = axes[1].imshow(outbound_grid, cmap='Oranges', interpolation='nearest')
        axes[1].set_title('Outbound Qubit Transfers')
        axes[1].invert_yaxis()
        plt.colorbar(im2, ax=axes[1], label='Outbound Traffic')

    # Total
    im3 = axes[-1].imshow(total_grid, cmap='hot', interpolation='nearest')
    axes[-1].set_title('Total Qubit Transfers (In + Out)')
    axes[-1].invert_yaxis()
    plt.colorbar(im3, ax=axes[-1], label='Total Traffic')

    plt.tight_layout()
    plt.savefig("routing_heatmaps_in_out_total.png", dpi=300, bbox_inches='tight')
    print("Saved inbound/outbound/total heatmaps as routing_heatmaps_in_out_total.png")
   
def compareplots():
    import time, csv, numpy as np, matplotlib.pyplot as plt, os
    from HqaPlacer import hqa_placer
    from PrevHqaPlacer import prev_hqa_placer
    from AppHqa import apphqa_placer
    from pytket import qasm

    # === Setup architecture and folders ===
    circuit_folders = [
        'circuits/large',
    ]
    repeats = 3
    out_csv = 'placer_comparison_with_qubo.csv'
    plot_file = 'hqa_vs_qubo.png'

    # Initialize placers
    hqa_inst = hqa_placer()
    phqa_inst = prev_hqa_placer()
    bfs_inst = bfs_placer()
    random_inst = random_placer()
    apphqa_placer_inst = apphqa_placer()


    # Create multicore architecture (adjust sizes/weights if needed)
    try:
        multicore_arch = multicore(4, 4, ncm_qubits=4, ncp_qubits=12, edge_weight_range=(1.5, 3.0))
    except Exception as e:
        print("Warning: multicore init failed with args, falling back to default ctor:", e)
        multicore_arch = multicore(4, 4)

    # Collect QASM files
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

    results = []
    for qfn in qasm_list:
        print(f"\n=== Processing {qfn} ===")
        circuit = qasm.circuit_from_qasm(qfn, maxwidth=1024)
        file_name = os.path.basename(qfn)

        entry = {'file': len(list(circuit))}

        # Helper to run a placer safely
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
                    # record a big cost and time to indicate failure
                    times.append(float('inf'))
                    costs.append(float('inf'))
            # Convert inf means to large numbers if needed, keep as-is for CSV
            mean_time = float(np.mean([t for t in times if np.isfinite(t)]) if any(np.isfinite(t) for t in times) else float('inf'))
            mean_cost = float(np.mean([c for c in costs if np.isfinite(c)]) if any(np.isfinite(c) for c in costs) else float('inf'))
            return mean_time, mean_cost



        # HQA Placer
        ht, hc = run_placer_n_times(hqa_inst, 'HQA', run_fn_name='place_per_timeslice')
        entry.update({'hqa_time_mean': ht, 'hqa_cost_mean': hc})
        print(f"HQA avg time={ht:.4f}s, cost={hc:.4f}")


        #random
        ht, hc = run_placer_n_times(random_inst, 'random', run_fn_name='place_per_timeslice')
        entry.update({'random_time_mean': ht, 'random_cost_mean': hc})
        print(f"random avg time={ht:.4f}s, cost={hc:.4f}")


        # bfs
        ht, hc = run_placer_n_times(bfs_inst, 'bfs', run_fn_name='place_per_timeslice')
        entry.update({'bfs_time_mean': ht, 'bfs_cost_mean': hc})
        print(f"bfs avg time={ht:.4f}s, cost={hc:.4f}")

        # pHQA Placer
        at, ac = run_placer_n_times(phqa_inst, 'Vanilla HQA', run_fn_name='place_per_timeslice')
        entry.update({'phqa_time_mean': at, 'phqa_cost_mean': ac})
        print(f"pHQA avg time={at:.4f}s, cost={ac:.4f}")

        # AppHQA Placer
        at, ac = run_placer_n_times(apphqa_placer_inst, 'app HQA', run_fn_name='place_per_timeslice')
        entry.update({'apphqa_time_mean': at, 'apphqa_cost_mean': ac})
        print(f"appHQA avg time={at:.4f}s, cost={ac:.4f}")


        results.append(entry)

    # === Save CSV ===
    with open(out_csv, 'w', newline='') as f:
        cols = ['file',
                'hqa_time_mean','hqa_cost_mean',
                'phqa_time_mean','phqa_cost_mean',
                'apphqa_time_mean','apphqa_cost_mean',
                'random_time_mean','random_cost_mean',
                'bfs_time_mean','bfs_cost_mean',
                ]
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        for r in results:
            # ensure all columns present
            for c in cols:
                if c not in r:
                    r[c] = float('inf')
            writer.writerow(r)
    print(f"\nWrote all results to {out_csv}")

    # === Plot comparison ===
    files = [r['file'] for r in results]
    order = np.argsort(files)
    files = [files[i] for i in order]
    hqa_sorted   = [results[i]['hqa_time_mean'] for i in order]
    apphqa_sorted  = [results[i]['apphqa_time_mean'] for i in order]
    phqa_sorted = [results[i]['phqa_time_mean'] for i in order]
    bfs_sorted = [results[i]['bfs_time_mean'] for i in order]
    random_sorted = [results[i]['random_time_mean'] for i in order]
    x = np.arange(len(files))
    plt.figure(figsize=(14,6))

    # Runtime subplot
    plt.subplot(1,2,1)
    plt.title('Average Runtime in s')
    width = 0.12
    offsets = [-1.5, -0.5, 0.5, 1.5, 2.5]
    plt.bar(x+offsets[1]*width, hqa_sorted, width=width, label='HQA')
    plt.bar(x+offsets[2]*width, phqa_sorted, width=width, label='Vanilla HQA')
    plt.bar(x+offsets[3]*width, apphqa_sorted, width=width, label='appHQA')
    plt.bar(x+offsets[4]*width, bfs_sorted, width=width, label='bfs')
    plt.bar(x+offsets[5]*width, random_sorted, width=width, label='random')
    plt.xticks(x, files, rotation=45, ha='right')
    plt.xlabel("#gates")
    plt.ylabel("#time")
    # plt.yscale('log')
    plt.legend()

    # Cost subplot (log scale)
    plt.subplot(1,2,2)
    plt.title('Average Cost')
    hqa_sorted   = [results[i]['hqa_cost_mean'] for i in order]
    apphqa_sorted  = [results[i]['apphqa_cost_mean'] for i in order]
    phqa_sorted = [results[i]['phqa_cost_mean'] for i in order]
    bfs_sorted = [results[i]['bfs_cost_mean'] for i in order]
    random_sorted = [results[i]['random_cost_mean'] for i in order]
    plt.bar(x+offsets[1]*width, hqa_sorted, width=width, label='HQA')
    plt.bar(x+offsets[2]*width, phqa_sorted, width=width, label=' Vanilla HQA')
    plt.bar(x+offsets[3]*width, apphqa_sorted, width=width, label='appHQA')
    plt.bar(x+offsets[4]*width, bfs_sorted, width=width, label='bfs')
    plt.bar(x+offsets[5]*width, random_sorted, width=width, label='random')
    plt.xticks(x, files, rotation=45, ha='right')
    plt.xlabel("#gates")
    plt.ylabel("#expected intercore communication")
    # plt.yscale('log')
    plt.legend()

    plt.tight_layout()
    plt.savefig(plot_file, dpi=300, bbox_inches='tight')
    print(f"Saved comparison plot to {plot_file}")
    print("\nDone. Results summary:")
    for r in results:
        print(r)






