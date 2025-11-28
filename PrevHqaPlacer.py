import numpy as np
import networkx as nx
import random
from collections import deque
from pytket import Circuit, OpType
from Placer import BasePlacer
from scipy.optimize import linear_sum_assignment as _hungarian

class prev_hqa_placer(BasePlacer):
    def place_per_timeslice(self, circuit, multicore_arch):
        """
        Hungarian Qubit Assignment (HQA) algorithm for placing qubits per timeslice
        Now uses weighted edge distances cached in a precomputed matrix for cost calculations
        (keeps Hungarian matching but uses cached distances & harmonized occupancy checks).
        """
        
        timeslices = self.break_into_timeslices(circuit)
        num_cores = len(multicore_arch.qpus)
        qubits_per_core = multicore_arch.qpu_qubit_num  # Assuming uniform core sizes
        core_nodes = list(multicore_arch.network.nodes) # Get the tuple-based nodes
        # Get all qubits in the circuit
        all_qubits = list(circuit.qubits)
        total_qubits = len(all_qubits)
        
        # Validate that we have enough space
        if total_qubits > num_cores * qubits_per_core:
            raise ValueError(f"Not enough space: {total_qubits} qubits need {num_cores * qubits_per_core} total slots")
        
        partition = []

        # For first timeslice, create initial round-robin assignment
        initial_assignment = {}
        for i, qubit in enumerate(all_qubits):
            core_id = int(i / qubits_per_core)
            core_node = core_nodes[core_id]
            initial_assignment[qubit] = core_node
        assert len(initial_assignment) == total_qubits, "Initial assignment incomplete"
        partition.append(initial_assignment)
        
        # ---------- Precompute weighted distance matrix (cache) ----------
        # Create node_index map and distance matrix to avoid repeated python calls
        self.node_index = {node: i for i, node in enumerate(core_nodes)}
        dist_matrix = np.zeros((num_cores, num_cores))
        self._dist_matrix = dist_matrix
        for i, n1 in enumerate(core_nodes):
            for j, n2 in enumerate(core_nodes):
                if i != j:
                    # rely on existing get_weighted_distance for computation but store result
                    dist_matrix[i, j] = self.get_weighted_distance(multicore_arch, n1, n2)
        dist_matrix[np.isnan(dist_matrix)] = np.inf

        # Process each subsequent timeslice
        for t in range(1, len(timeslices)):
            current_timeslice = timeslices[t]
            previous_assignment = partition[t-1].copy()
            
            # Start with previous assignment - ensures no qubit is lost
            new_assignment = previous_assignment.copy()
            
            # Step 1: Identify unfeasible two-qubit gates
            unfeasible_operations = []
            two_qubit_gates = [gate for gate in current_timeslice if len(gate.qubits) == 2 
                            and gate.op.type not in [OpType.Measure, OpType.Reset]]
            
            for gate in two_qubit_gates:
                q1, q2 = gate.qubits[0], gate.qubits[1]
                if previous_assignment[q1] != previous_assignment[q2]:
                    unfeasible_operations.append((gate, q1, q2))
            
            if not unfeasible_operations:
                # No unfeasible operations, keep previous assignment
                partition.append(new_assignment)
                continue
            
            # Step 2: Track core occupancy
            core_occupancy = {node: 0 for node in core_nodes}
            for qubit, core_node in new_assignment.items():
                core_occupancy[core_node] += 1
            
            # Step 3: Process unfeasible operations iteratively
            remaining_operations = unfeasible_operations.copy()
            max_iterations = len(unfeasible_operations) + 1  # same guard as AppHqa
            iteration = 0
            
            while remaining_operations and iteration < max_iterations:
                iteration += 1
                
                # Find operations we can assign (cores with enough space)
                assignable_operations = []
                for op in remaining_operations:
                    gate, q1, q2 = op
                    # Check if we can find a core for this operation
                    found_assignable = False
                    for core_node in core_nodes:
                        # Harmonized check (same threshold as AppHqa)
                        if core_occupancy[core_node] <= qubits_per_core - 2:
                            found_assignable = True
                            break
                    if found_assignable:
                        assignable_operations.append(op)
                
                if not assignable_operations:
                    # No operations can be assigned, need to make space
                    # Move some qubits to create space
                    self._make_space_for_operations(remaining_operations, new_assignment, 
                                                core_occupancy, qubits_per_core, core_nodes, multicore_arch)
                    continue
                
                # Create cost matrix for assignable operations
                num_ops = len(assignable_operations)
                available_cores = [node for node in core_nodes 
                                if core_occupancy[node] <= qubits_per_core - 2]
                
                if not available_cores:
                    break
                
                # Build cost matrix using cached distance matrix where possible
                cost_matrix = np.full((num_ops, len(available_cores)), float('inf'))
                
                # Use local references to speed up repeated lookups
                local_node_index = self.node_index
                local_dist = self._dist_matrix
                for op_idx, (gate, q1, q2) in enumerate(assignable_operations):
                    # current cores for q1 and q2 (if any)
                    q1_core = new_assignment.get(q1, None)
                    q2_core = new_assignment.get(q2, None)
                    for core_idx, core_node in enumerate(available_cores):
                        if core_occupancy[core_node] <= qubits_per_core - 2:
                            
                            total_cost = self._calculate_operation_cost(q1, q2, core_node, new_assignment, timeslices, t, multicore_arch)
                            
                            cost_matrix[op_idx, core_idx] = max(0.1, total_cost)
                
                # Apply Hungarian algorithm
                if num_ops > 0 and len(available_cores) > 0:
                    try:
                        row_indices, col_indices = _hungarian(cost_matrix)
                        
                        # Apply valid assignments
                        assigned_ops = []
                        for op_idx, core_idx in zip(row_indices, col_indices):
                            if (op_idx < num_ops and core_idx < len(available_cores) and 
                                cost_matrix[op_idx, core_idx] != float('inf')):
                                
                                gate, q1, q2 = assignable_operations[op_idx]
                                core_node = available_cores[core_idx]
                                
                                # Remove qubits from old cores
                                if q1 in new_assignment:
                                    core_occupancy[new_assignment[q1]] -= 1
                                if q2 in new_assignment:
                                    core_occupancy[new_assignment[q2]] -= 1
                                
                                # Assign to new core
                                new_assignment[q1] = core_node
                                new_assignment[q2] = core_node
                                core_occupancy[core_node] += 2
                                
                                assigned_ops.append(assignable_operations[op_idx])
                        
                        # Remove assigned operations
                        remaining_operations = [op for op in remaining_operations 
                                            if op not in assigned_ops]
                        
                            
                    except Exception as e:

                        break
            

            
            # Step 5: Ensure all original qubits are still assigned
            for qubit in all_qubits:
                if qubit not in new_assignment:
                    # Find least loaded core and assign
                    min_core = min(core_nodes, key=lambda c: core_occupancy[c])
                    if core_occupancy[min_core] < qubits_per_core:
                        new_assignment[qubit] = min_core
                        core_occupancy[min_core] += 1
                    else:
                        # Emergency: find any core with space
                        for core_node in core_nodes:
                            if core_occupancy[core_node] < qubits_per_core:
                                new_assignment[qubit] = core_node
                                core_occupancy[core_node] += 1
                                break
            
            # Final verification
            assert len(new_assignment) == total_qubits, f"Assignment incomplete: {len(new_assignment)}/{total_qubits}"
            assert all(qubit in new_assignment for qubit in all_qubits), "Some qubits unassigned"
            
            partition.append(new_assignment)
        
        return partition

    
    
    def _make_space_for_operations(self, operations, assignment, core_occupancy, qubits_per_core, core_nodes, multicore_arch):
        """
        Make space in cores by moving qubits around using weighted distance-based costs
        """
        # Find overcrowded cores
        for core_node in core_nodes:
            if core_occupancy[core_node] > qubits_per_core - 2:
                # Find qubits to move from this core
                qubits_in_core = [q for q, c in assignment.items() if c == core_node]
                
                # Try to move some qubits to less crowded cores, prioritizing closer cores (by weighted distance)
                for qubit in qubits_in_core[:2]:  # Move up to 2 qubits
                    # Sort target cores by weighted distance from current core and available space
                    target_cores = []
                    for target_core in core_nodes:
                        if core_occupancy[target_core] < qubits_per_core - 1 and target_core != core_node:
                            weighted_distance = self.get_weighted_distance(multicore_arch, core_node, target_core)
                            target_cores.append((target_core, weighted_distance, core_occupancy[target_core]))
                    
                    # Sort by weighted distance first, then by occupancy
                    target_cores.sort(key=lambda x: (x[1], x[2]))
                    
                    for target_core, distance, occupancy in target_cores:
                        assignment[qubit] = target_core
                        core_occupancy[core_node] -= 1
                        core_occupancy[target_core] += 1
                        break
                    
                    if core_occupancy[core_node] <= qubits_per_core - 2:
                        break

    

    def _calculate_operation_cost(self, q1, q2, core_node, current_assignment, timeslices, current_t, multicore_arch):
        """
        Calculate cost of assigning an operation to a core using weighted distance-based costs
        """
        # Weighted dynamic cost based on network distance for qubit movements
        movement_cost = 0
        
        # Cost for moving q1 to target core (using weighted distance)
        if q1 in current_assignment and current_assignment[q1] != core_node:
            movement_cost+=1
        elif q1 not in current_assignment:
            movement_cost += 1  # Base cost for unassigned qubit
        
        # Cost for moving q2 to target core (using weighted distance)
        if q2 in current_assignment and current_assignment[q2] != core_node:
            movement_cost+=1
        elif q2 not in current_assignment:
            movement_cost += 1  # Base cost for unassigned qubit
        
        # Future interaction bonus (negative cost) - weighted by distances
        attraction = 0.0
        max_lookahead = min(current_t + 4, len(timeslices))
        
        # for t in range(current_t + 1, max_lookahead):
        #     weight = 2 ** -(t - current_t)
        #     timeslice = timeslices[t]
            
        #     for gate in timeslice:
        #         if len(gate.qubits) == 2:
        #             gate_q1, gate_q2 = gate.qubits[0], gate.qubits[1]
                    
        #             # Check interactions with qubits already in target core
        #             for assigned_qubit, assigned_core in current_assignment.items():
        #                 if assigned_core == core_node:
        #                     if ((gate_q1 == q1 and gate_q2 == assigned_qubit) or
        #                         (gate_q2 == q1 and gate_q1 == assigned_qubit) or
        #                         (gate_q1 == q2 and gate_q2 == assigned_qubit) or
        #                         (gate_q2 == q2 and gate_q1 == assigned_qubit)):
        #                         # Weight attraction by inverse of average weighted distance in the network
        #                         # This makes cores with lower-weight connections more attractive for future interactions
        #                         avg_weighted_distance = self._get_average_weighted_distance(multicore_arch)
        #                         distance_weight = 1.0 / (avg_weighted_distance + 0.1)
        #                         attraction += weight * distance_weight
        
        for t in range(current_t + 1, min(len(timeslices), max_lookahead)):
            weight = 2 ** -(t - current_t)
            for gate in timeslices[t]:
                if len(gate.qubits) == 2:
                    fq1, fq2 = gate.qubits
                    if fq1 in (q1, q2) or fq2 in (q1, q2):
                        other = fq2 if fq1 in (q1, q2) else fq1
                        if other in current_assignment and current_assignment[other] == core_node:
                            attraction += weight * 0.9
                        else:
                            attraction += weight * 0.3
        
        # Total cost = movement cost - attraction bonus
        total_cost = movement_cost - attraction
        
        return max(0.1, total_cost)  # Ensure minimum positive cost

    def _get_average_weighted_distance(self, multicore_arch):
        """
        Calculate average weighted distance in the network topology
        """
        nodes = list(multicore_arch.network.nodes)
        total_weighted_distance = 0
        pairs_count = 0
        
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                weighted_distance = self.get_weighted_distance(multicore_arch, nodes[i], nodes[j])
                if weighted_distance != float('inf'):
                    total_weighted_distance += weighted_distance
                    pairs_count += 1
        
        return total_weighted_distance / pairs_count if pairs_count > 0 else 1.0

    def per_timeslice_cost(self, circuit, partition, multicore_arch):
        """
        Calculates the total inter-core communication cost for a given partition using weighted distances.
        """
        total_communication_cost = 0
        timeslices = self.break_into_timeslices(circuit)
        
        # Communication cost for two-qubit gates (using weighted distances)
        for i, timeslice in enumerate(timeslices):
            current_assignment = partition[i]
            for gate in timeslice:
                # Only consider two-qubit gates that are not Measure or Reset
                if len(gate.qubits) == 2 and gate.op.type not in [OpType.Measure, OpType.Reset]:
                    q1, q2 = gate.qubits[0], gate.qubits[1]
                    # Check if qubits are on different cores
                    if current_assignment[q1] != current_assignment[q2]:
                        # assert False, "Two-qubit gate assigned to different cores in partition!"
                        weighted_distance = self.get_weighted_distance(multicore_arch, 
                                                                      current_assignment[q1], 
                                                                      current_assignment[q2])
                        total_communication_cost += weighted_distance

        # Communication cost for qubit movements between timeslices (using weighted distances)
        for i in range(2, len(timeslices)):
            prev_assignment = partition[i-1]
            curr_assignment = partition[i]
            for qubit in curr_assignment:
                # If qubit's core has changed
                if curr_assignment[qubit] != prev_assignment[qubit]:
                    weighted_distance = self.get_weighted_distance(multicore_arch,
                                                                  prev_assignment[qubit],
                                                                  curr_assignment[qubit])
                    total_communication_cost += weighted_distance

        return total_communication_cost
    def per_core_intercore_activity(self, circuit, partition, multicore_arch):
        """
        Calculate the total number of inter-core communications each core participates in
        across the entire circuit execution (all timeslices combined).

        Returns:
            total_comm_per_core: dict {core_node: total_intercore_count}
        """
        timeslices = self.break_into_timeslices(circuit)
        num_slices = len(timeslices)
        core_nodes = list(multicore_arch.network.nodes)

        # Initialize total communication count per core
        total_comm_per_core = {core: 0 for core in core_nodes}

        # --- Part 1: Two-qubit gate inter-core communications (within timeslices)
        for t in range(num_slices):
            current_assignment = partition[t]
            for gate in timeslices[t]:
                if len(gate.qubits) == 2 and gate.op.type not in [OpType.Measure, OpType.Reset]:
                    q1, q2 = gate.qubits
                    c1, c2 = current_assignment[q1], current_assignment[q2]
                    if c1 != c2:
                        # Both cores participated in inter-core communication
                        total_comm_per_core[c1] += 1
                        total_comm_per_core[c2] += 1

        # --- Part 2: Qubit movement between consecutive timeslices
        for t in range(1, num_slices):
            prev_assignment = partition[t - 1]
            curr_assignment = partition[t]
            for qubit in curr_assignment:
                c_prev, c_curr = prev_assignment[qubit], curr_assignment[qubit]
                if c_prev != c_curr:
                    # Both source and destination cores are involved
                    total_comm_per_core[c_prev] += 1
                    total_comm_per_core[c_curr] += 1

        return total_comm_per_core



    def simulate_broken_links(self, circuit, multicore_arch, num_simulations=10, break_probability=0.1):
        """
        Simulate 500 mappings with subsets of links broken (infinite weight) and calculate costs.
        
        Args:
            circuit: The quantum circuit to simulate.
            multicore_arch: The multicore architecture object.
            num_simulations: Number of mappings to simulate.
            break_probability: Probability of breaking each link in the network.

        Returns:
            A list of tuples (broken_links, cost), where:
                - broken_links: Set of edges marked as broken.
                - cost: Communication cost for the mapping.
        """
        results = []
        original_network = multicore_arch.network.copy()

        for _ in range(num_simulations):
            # Create a copy of the network and break some links
            simulated_network = original_network.copy()
            broken_links = set()
            for edge in original_network.edges:
                if random.random() < break_probability:
                    simulated_network[edge[0]][edge[1]]['weight'] = float('inf')
                    broken_links.add(edge)

            # Update the architecture with the simulated network
            multicore_arch.network = simulated_network

            # Calculate the cost for the current mapping
            partition = self.place_per_timeslice(circuit, multicore_arch)
            cost = self.per_timeslice_cost(circuit, partition, multicore_arch)

            # Store the result
            results.append((broken_links, cost))

        # Restore the original network
        multicore_arch.network = original_network

        print(results)
        return results