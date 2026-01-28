import numpy as np
import networkx as nx
from pytket import OpType
import matplotlib.pyplot as plt
from pyqubo import Array,Binary
import neal
from Placer import BasePlacer

class qubo_placer(BasePlacer):

    def place_per_timeslice(self, circuit, multicore_arch):
        """
        Per-timeslice placement using pyqubo + neal, modeled after optimization.py:
        - x_{j,i} : 1 if logical qubit i is placed on core j
        - y_{j,p} : slack vars enforcing sum_i x_{j,i} <= capacity
        - Assignment constraint: each logical qubit appears on exactly one core
        - Laplacian term: keeps edges of the interaction graph within a core
        - Movement penalty: discourages moving the same logical qubit between cores across slices
        """

        timeslices = self.break_into_timeslices(circuit)
        timeslice_partitions = []

        node_ids = list(multicore_arch.network.nodes())  # stable order of cores
        k = len(node_ids)  # Number of cores
        cap = multicore_arch.qpu_qubit_num  # Capacity of each core

        # weights from optimization.py
        A_assign = 4.0     # assignment
        B_capacity = 4.0   # capacity (with slack)
        C_laplacian = 10.0  # Keep connected qubits together, default weight
        D_movement = 10.0   # Penalize movement between slices

        # Precompute core-to-core shortest path distances
        core_dist_matrix = np.zeros((k, k))
        for j_idx, core_j in enumerate(node_ids):
            for l_idx, core_l in enumerate(node_ids):
                core_dist_matrix[j_idx, l_idx] = nx.shortest_path_length(multicore_arch.network, core_j, core_l)

        x = {}
        y = {}
        all_qubits_per_slice = []
        L_per_slice = []

        # First, create variables & store slice data
        for t_idx, timeslice in enumerate(timeslices):
            # Sort qubits to ensure stable indexing
            qubits = sorted({q for gate in timeslice for q in gate.qubits}, key=lambda q: str(q))
            all_qubits_per_slice.append(qubits)
            n = len(qubits) # Number of logical qubits in this slice

            # Build interaction graph and Laplacian
            G = nx.Graph()
            qubit_count = len(qubits)  # Let's use a new variable for clarity
            if qubit_count > 0:
                G.add_nodes_from(range(qubit_count))
                q_to_i = {q: i for i, q in enumerate(qubits)}
                for gate in timeslice:
                    if len(gate.qubits) > 1:
                        qs = gate.qubits
                        for a in range(len(qs)):
                            for b in range(a + 1, len(qs)):
                                ia = q_to_i[qs[a]]
                                ib = q_to_i[qs[b]]
                                if ia != ib:
                                    G.add_edge(ia, ib)
                L = nx.laplacian_matrix(G, nodelist=range(qubit_count)).toarray()
            else:
                L = np.zeros((0, 0)) # Explicitly create an empty matrix for the zero-qubit case

            L_per_slice.append(L)

            # Create Binary vars for this slice
            for j in range(k):
                x[(t_idx, j)] = [Binary(f'x_{t_idx}_{j}-{i}') for i in range(n)]
                y[(t_idx, j)] = [Binary(f'y_{t_idx}_{j}-{p}') for p in range(cap)]

        # Now build the objective based on optimization.py logic
        model_expr = 0
        n_slices = len(timeslices)


        # Sum over all slices
        for t_idx in range(n_slices):
            qubits = all_qubits_per_slice[t_idx]
            n = len(qubits)
            L = L_per_slice[t_idx]

            # 1. Assignment constraint (each qubit must be on exactly one core)
            if n > 0:
                term_assignment = sum(
                    [(sum(x[(t_idx, j)][i] for j in range(k)) - 1)**2 for i in range(n)]
                )
            else:
                term_assignment = 0

            # 2. Capacity constraint (core load <= capacity)
            term_capacity = sum(
                [(sum(x[(t_idx, j)]) - sum(y[(t_idx, j)]))**2 for j in range(k)]
            )

            # 3. Laplacian term (keep connected qubits together within a core)
            if n > 0:
                term_laplacian = sum([
                    sum(L[i, i] * x[(t_idx, j)][i] for i in range(n)) +
                    2 * sum(
                        sum(L[i, l_idx] * x[(t_idx, j)][i] * x[(t_idx, j)][l_idx]
                            for i in range(l_idx + 1, n))
                        for l_idx in range(n - 1)
                    )
                    for j in range(k)
                ])
            else:
                term_laplacian = 0

            model_expr += A_assign * term_assignment + B_capacity * term_capacity + C_laplacian * term_laplacian
            
            # # 4. Movement penalty term (between adjacent slices)
            if t_idx < n_slices - 1:
                next_qubits = all_qubits_per_slice[t_idx + 1]
                
                common_qs = [q for q in qubits if q in next_qubits]
                movement_term_slice = 0

                for q in common_qs:
                    i_cur = qubits.index(q)
                    i_next = next_qubits.index(q)
                    
                    for j in range(k):
                        for l in range(k):
                            movement_term_slice += core_dist_matrix[j, l] * x[(t_idx, j)][i_cur] * x[(t_idx + 1, l)][i_next]
                
                model_expr += D_movement * movement_term_slice

        # Compile and solve
        compiled = model_expr.compile()
        bqm = compiled.to_bqm()
        print("herebf\n")
        sampler = neal.SimulatedAnnealingSampler()
        sampleset = sampler.sample(bqm, num_reads=10, seed=42) # reduced num_reads for speed
        decoded = compiled.decode_sampleset(sampleset)
        best = min(decoded, key=lambda s: s.energy)
        sample = best.sample

        # Decode mapping per slice
        for t_idx, qubits in enumerate(all_qubits_per_slice):
            mapping = {}
            load = {j: 0 for j in range(k)}
            n = len(qubits)

            for i, q in enumerate(qubits):
                chosen_j = None
                for j in range(k):
                    # Correctly access variables from the sample
                    if sample.get(f"x_{t_idx}_{j}-{i}", 0) == 1:
                        chosen_j = j
                        break
                
                if chosen_j is not None and load[chosen_j] < cap:
                    mapping[q] = node_ids[chosen_j]
                    load[chosen_j] += 1
                else:
                    mapping[q] = None
            
            # Fix unassigned qubits by assigning to the least loaded core
            for i, q in enumerate(qubits):
                if mapping[q] is None:
                    best_core_idx = min(range(k), key=lambda jj: load[jj])
                    mapping[q] = node_ids[best_core_idx]
                    load[best_core_idx] += 1

            timeslice_partitions.append(mapping)

        return timeslice_partitions


    def per_timeslice_cost(self, circuit, partition, multicore_arch):
        total_communication_cost = 0
        withints =0
        interts =0
        qubit_last_qpu = {}        
        timeslices = self.break_into_timeslices(circuit)
        print(len(timeslices),len(partition))
        for i in range(len(timeslices)):
            timeslice = timeslices[i]
            for gate in timeslice:
                type = gate.op.type
                qubits = gate.qubits
                if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:

                    if partition[i][qubits[0]] != partition[i][qubits[1]]:
                        distance = nx.shortest_path_length(multicore_arch.network, partition[i][qubits[0]], partition[i][qubits[1]])
                        total_communication_cost += distance
                        withints+=distance

                for qubit in qubits:
                    current_qpu = partition[i].get(qubit)
                    last_qpu = qubit_last_qpu.get(qubit)
                    
                    if last_qpu is not None and current_qpu != last_qpu:
                        val = nx.shortest_path_length(multicore_arch.network, current_qpu,last_qpu)
                        total_communication_cost +=val
                        interts +=val
                        
                    qubit_last_qpu[qubit] = current_qpu
        print(withints,interts,"heha")           
        return total_communication_cost
    def intercorecommatrix(self,circuit,partition,multicore_arch):

        qubit_last_qpu = {}
        timeslices = self.break_into_timeslices(circuit)

        # Dimensions of the QPU grid (flattened index space)
        num_qpus = len(multicore_arch.network.nodes())
        comm_matrix = np.zeros((num_qpus, num_qpus), dtype=int)

        # Go through each timeslice
        for i in range(len(timeslices)):
            timeslice = timeslices[i]
            for gate in timeslice:
                type = gate.op.type
                qubits = gate.qubits

                # Two-qubit gates between different QPUs → intercore communication
                if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                    qpu_a = partition[i][qubits[0]]
                    qpu_b = partition[i][qubits[1]]
                    if qpu_a != qpu_b:
                        idx_a = list(multicore_arch.network.nodes()).index(qpu_a)
                        idx_b = list(multicore_arch.network.nodes()).index(qpu_b)
                        comm_matrix[idx_a, idx_b] += 1
                        comm_matrix[idx_b, idx_a] += 1  # symmetric

                # Track qubit movement between timeslices
                for qubit in qubits:
                    current_qpu = partition[i].get(qubit)
                    last_qpu = qubit_last_qpu.get(qubit)

                    if last_qpu is not None and current_qpu != last_qpu:
                        idx_a = list(multicore_arch.network.nodes()).index(current_qpu)
                        idx_b = list(multicore_arch.network.nodes()).index(last_qpu)
                        comm_matrix[idx_a, idx_b] += 1
                        comm_matrix[idx_b, idx_a] += 1  # symmetric

                    qubit_last_qpu[qubit] = current_qpu

        # Plot and save heatmap
        plt.figure(figsize=(8, 6))
        plt.imshow(comm_matrix, cmap='hot', interpolation='nearest')
        plt.colorbar(label="Intercore Communication Count")
        plt.title("Intercore Communication Matrix")
        plt.xlabel("QPU index")
        plt.ylabel("QPU index")
        plt.savefig("qubointercorecommunicationcount.png", bbox_inches='tight')
        plt.close()


    def distribution_matrix(self, circuit, partition, multicore_arch):


        timeslices = self.break_into_timeslices(circuit)

        num_qpus = len(multicore_arch.network.nodes())
        dist_matrix = np.zeros((num_qpus,), dtype=int)  # count per QPU

        # Go through each timeslice and count assignments
        for i in range(len(timeslices)):
            assigned_qpus = partition[i].values()  # QPU coords where each qubit is placed
            for qpu_coord in assigned_qpus:
                idx = list(multicore_arch.network.nodes()).index(qpu_coord)
                dist_matrix[idx] += 1
        for i in range(len(multicore_arch.network.nodes())):
            dist_matrix[i] /= len(timeslices)
        # Reshape into 2D grid for plotting (optional, based on architecture layout)
        side_x = len({coord[0] for coord in multicore_arch.network.nodes()})
        side_y = len({coord[1] for coord in multicore_arch.network.nodes()})
        dist_grid = dist_matrix.reshape((side_x, side_y))

        # Plot and save heatmap
        plt.figure(figsize=(6, 5))
        plt.imshow(dist_grid, cmap='Blues', interpolation='nearest')
        plt.colorbar(label="Qubit Assignment Count")
        plt.title("Qubit Distribution Across Cores")
        plt.xlabel("QPU X-coordinate")
        plt.ylabel("QPU Y-coordinate")
        plt.savefig("qubit_distribution_matrix.png", bbox_inches='tight')
        plt.close()


    def visualize_qubit_distribution(self, timeslice_partitions, multicore_arch):
        """
        Visualizes the distribution of logical qubits across physical cores for each timeslice.
        The resulting plot is saved to a file instead of being displayed interactively.
        """
        num_slices = len(timeslice_partitions)
        core_ids = list(multicore_arch.network.nodes())

        # Determine the grid size for subplots
        cols = 3  # You can adjust this for a better layout
        rows = (num_slices + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))

        # Check if axes is a single Axes object or an array
        if num_slices == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        for i, partition in enumerate(timeslice_partitions):
            ax = axes[i]
            qubit_counts = {core: 0 for core in core_ids}
            for core in partition.values():
                qubit_counts[core] += 1

            # Convert core IDs (which might be tuples) to strings for clean plotting
            cores_labels = [str(core) for core in qubit_counts.keys()]
            counts = list(qubit_counts.values())

            ax.bar(cores_labels, counts)
            ax.set_title(f'Timeslice {i}')
            ax.set_xlabel('Physical Core')
            ax.set_ylabel('Number of Logical Qubits')
            ax.set_ylim(0, multicore_arch.qpu_qubit_num)

        # Hide any unused subplots
        for i in range(num_slices, len(axes)):
            axes[i].axis('off')

        plt.tight_layout()
        
        # Save the figure to a file instead of showing it
        plt.savefig('qubit_distribution_all_timeslices.png')
        plt.close(fig) # Close the figure to free up memory
        
    def visualize_intercore_communication_per_layer(self, circuit, multicore_arch):
        """
        Calculates and visualizes the inter-core communication cost for each timeslice.
        The resulting plot is saved to a file instead of being displayed interactively.
        """
        timeslices = self.break_into_timeslices(circuit)
        timeslice_partitions = self.place_per_timeslice(circuit, multicore_arch)
        
        num_slices = len(timeslices)
        communication_costs = []
        qubit_last_qpu = {}

        for i in range(num_slices):
            timeslice_cost = 0
            timeslice = timeslices[i]
            partition = timeslice_partitions[i]
            
            # Calculate cost for two-qubit gates
            for gate in timeslice:
                if len(gate.qubits) == 2 and gate.op.type not in (OpType.Measure, OpType.Reset):
                    q1, q2 = gate.qubits[0], gate.qubits[1]
                    if partition.get(q1) != partition.get(q2):
                        distance = nx.shortest_path_length(multicore_arch.network, partition.get(q1), partition.get(q2))
                        timeslice_cost += distance

            # Calculate cost for qubit movement
            for qubit in partition.keys():
                current_qpu = partition.get(qubit)
                last_qpu = qubit_last_qpu.get(qubit)

                if last_qpu is not None and current_qpu != last_qpu:
                    timeslice_cost += nx.shortest_path_length(multicore_arch.network, current_qpu, last_qpu)

            # Update qubit_last_qpu for the next timeslice
            for qubit, qpu in partition.items():
                qubit_last_qpu[qubit] = qpu

            communication_costs.append(timeslice_cost)

        # Create and save the bar chart
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.bar(range(num_slices), communication_costs)
        ax.set_title('Inter-core Communication Cost Per Timeslice')
        ax.set_xlabel('Timeslice Index')
        ax.set_ylabel('Communication Cost')
        
        plt.tight_layout()
        plt.savefig('intercore_communication_per_layer.png')
        plt.close(fig)