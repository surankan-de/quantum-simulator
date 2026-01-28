from Placer import BasePlacer
from pytket import OpType
import random

def random_initial_assignment(all_qubits, core_nodes, qubits_per_core, seed=None):
    if seed is not None:
        random.seed(seed)

    core_slots = []
    for c in core_nodes:
        core_slots.extend([c] * qubits_per_core)

    random.shuffle(core_slots)

    return dict(zip(all_qubits, core_slots))

class gtrhqa_placer(BasePlacer):

    def place_per_timeslice(self, circuit, multicore_arch):
        """
        Optimized Approximate Hungarian Qubit Assignment (HQA) algorithm.
        Replaces the cubic Hungarian step with a vectorized greedy O(n²) step.
        Includes:
        - caching of weighted distances
        - vectorized cost matrix computation
        - minimal dictionary copying
        - reduced recomputation of operation costs
        """
        import numpy as np
        from pytket import OpType

        timeslices = self.break_into_timeslices(circuit)
        num_cores = len(multicore_arch.qpus)
        qubits_per_core = multicore_arch.qpu_qubit_num
        core_nodes = list(multicore_arch.network.nodes)

        all_qubits = list(circuit.qubits)
        total_qubits = len(all_qubits)

        if total_qubits > num_cores * qubits_per_core:
            raise ValueError(f"Not enough space: {total_qubits} qubits > capacity {num_cores * qubits_per_core}")

        partition = []

        initial_assignment = {q: core_nodes[int(i / qubits_per_core)] for i, q in enumerate(all_qubits)}

        partition.append(initial_assignment)

        dist_matrix = np.zeros((num_cores, num_cores))
        self._dist_matrix = dist_matrix
        self.node_index = {node: i for i, node in enumerate(core_nodes)}
        for i, n1 in enumerate(core_nodes):
            for j, n2 in enumerate(core_nodes):
                if i != j:
                    dist_matrix[i, j] = self.get_weighted_distance(multicore_arch, n1, n2)
        dist_matrix[np.isnan(dist_matrix)] = np.inf  
             


        for t in range(1, len(timeslices)):
            current_timeslice = timeslices[t]
            if(t>0):

                prev_assignment = partition[t - 1]
            else:
                prev_assignment = initial_assignment
            new_assignment = dict(prev_assignment)

            two_qubit_gates = [
                g for g in current_timeslice if len(g.qubits) == 2 and g.op.type not in [OpType.Measure, OpType.Reset]
            ]
            unfeasible_ops = [
                (g, g.qubits[0], g.qubits[1])
                for g in two_qubit_gates
                if prev_assignment[g.qubits[0]] != prev_assignment[g.qubits[1]]
            ]

            if not unfeasible_ops:
                partition.append(new_assignment)
                continue

            core_occupancy = {node: 0 for node in core_nodes}
            for q, c in new_assignment.items():
                core_occupancy[c] += 1

            available_cores = [node for node in core_nodes if core_occupancy[node] <= qubits_per_core - 2]
            if len(available_cores) == 0:
                partition.append(new_assignment)
                continue

            num_ops = len(unfeasible_ops)
            num_avail = len(available_cores)

            assignments = new_assignment
            get_dist = lambda n1, n2: self._dist_matrix[self.node_index[n1], self.node_index[n2]]
            cost_matrix = np.empty((num_ops, num_avail))
            for i, (g, q1, q2) in enumerate(unfeasible_ops):
                q1_core = assignments.get(q1)
                q2_core = assignments.get(q2)
                for j, cnode in enumerate(available_cores):
                    cost = 0
                    if q1_core and q1_core != cnode:
                        cost += get_dist(q1_core, cnode)
                    elif q1_core is None:
                        cost += 1
                    if q2_core and q2_core != cnode:
                        cost += get_dist(q2_core, cnode)
                    elif q2_core is None:
                        cost += 1
                    cost_matrix[i, j] = cost


            avail_count = len(available_cores)
            num_ops = len(unfeasible_ops)

            occupancy_arr = np.array([core_occupancy[c] for c in available_cores], dtype=np.float32)

            if num_ops == 0 or avail_count == 0:
                pass
            else:
                assigned_op_for_core = -np.ones(avail_count, dtype=np.int32)  
                assigned_core_for_op = -np.ones(num_ops, dtype=np.int32)     
                core_available_mask = np.ones(avail_count, dtype=bool)

                finite_mask = np.isfinite(cost_matrix)

                ops_remaining_mask = np.ones(num_ops, dtype=bool)
                max_rounds = min(avail_count, num_ops) + 2
                round_i = 0

                while ops_remaining_mask.any() and core_available_mask.any() and round_i < max_rounds:
                    round_i += 1

                    cols_idx = np.nonzero(core_available_mask)[0]
                    if cols_idx.size == 0:
                        break

                    rows_idx = np.nonzero(ops_remaining_mask)[0]
                    sub = cost_matrix[rows_idx][:, cols_idx]

                    if not np.isfinite(sub).any():
                        break

                    huge = 1e12
                    sub_masked = np.where(np.isfinite(sub), sub, huge)
                    rel_col = np.argmin(sub_masked, axis=1)                   
                    chosen_costs = sub_masked[np.arange(sub_masked.shape[0]), rel_col]
                    chosen_cores = cols_idx[rel_col]                          
                    chosen_ops = rows_idx                                    
                    unique_cores, inv = np.unique(chosen_cores, return_inverse=True)
                    picks_rows = []
                    picks_cores = []
                    for ui, core_idx in enumerate(unique_cores):
                        idxs = np.where(inv == ui)[0]  
                        if idxs.size == 0:
                            continue
                        local_costs = chosen_costs[idxs]
                        best_local_i = idxs[int(np.argmin(local_costs))]
                        op_to_assign = int(chosen_ops[best_local_i])  
                        picks_rows.append(op_to_assign)
                        picks_cores.append(int(core_idx))

                    for op_idx, core_idx in zip(picks_rows, picks_cores):
                        if assigned_core_for_op[op_idx] != -1:
                            continue
                        if not core_available_mask[core_idx]:
                            continue
                        cost_val = cost_matrix[op_idx, core_idx]
                        if not np.isfinite(cost_val):
                            continue

                        assigned_core_for_op[op_idx] = core_idx
                        assigned_op_for_core[core_idx] = op_idx
                        core_available_mask[core_idx] = False
                        ops_remaining_mask[op_idx] = False

                        gate, q1, q2 = unfeasible_ops[op_idx]
                        core_node = available_cores[core_idx]

                        if q1 in new_assignment:
                            core_occupancy[new_assignment[q1]] -= 1
                        if q2 in new_assignment:
                            core_occupancy[new_assignment[q2]] -= 1

                        new_assignment[q1] = core_node
                        new_assignment[q2] = core_node

                        occupancy_arr[core_idx] += 2


                remaining_ops = np.where(assigned_core_for_op == -1)[0]
                remaining_core_idxs = np.where(core_available_mask)[0].tolist()
                if remaining_ops.size > 0 and len(remaining_core_idxs) > 0:
                    for op_idx in remaining_ops:
                        if not remaining_core_idxs:
                            break
                        costs_for_op = cost_matrix[op_idx, remaining_core_idxs]
                        finite_mask_local = np.isfinite(costs_for_op)
                        if not finite_mask_local.any():
                            continue
                        rel = int(np.argmin(np.where(finite_mask_local, costs_for_op, 1e12)))
                        chosen_core_rel = remaining_core_idxs[rel]
                        gate, q1, q2 = unfeasible_ops[op_idx]
                        core_node = available_cores[chosen_core_rel]

                        if q1 in new_assignment:
                            core_occupancy[new_assignment[q1]] -= 1
                        if q2 in new_assignment:
                            core_occupancy[new_assignment[q2]] -= 1

                        new_assignment[q1] = core_node
                        new_assignment[q2] = core_node

                        occupancy_arr[chosen_core_rel] += 2
                        try:
                            remaining_core_idxs.remove(chosen_core_rel)
                        except ValueError:
                            pass



            # Ensure every qubit is assigned
            for q in all_qubits:
                if q not in new_assignment:
                    min_core = min(core_nodes, key=lambda c: core_occupancy[c])
                    new_assignment[q] = min_core
                    core_occupancy[min_core] += 1

            partition.append(new_assignment)

        return partition


    def _make_space_for_operations(self, operations, assignment, core_occupancy, qubits_per_core, core_nodes, multicore_arch):
        """
        Make space in cores by moving qubits around using weighted distance-based costs
        """
        for core_node in core_nodes:
            if core_occupancy[core_node] > qubits_per_core - 2:
                qubits_in_core = [q for q, c in assignment.items() if c == core_node]
                for qubit in qubits_in_core[:2]:
                    target_cores = []
                    for target_core in core_nodes:
                        if core_occupancy[target_core] < qubits_per_core - 1 and target_core != core_node:
                            weighted_distance = self.get_weighted_distance(multicore_arch, core_node, target_core)
                            target_cores.append((target_core, weighted_distance, core_occupancy[target_core]))
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
        get_dist = lambda n1, n2: self._dist_matrix[self.node_index[n1], self.node_index[n2]]
        movement_cost = 0
        if q1 in current_assignment and current_assignment[q1] != core_node:
            weighted_distance_q1 = get_dist(current_assignment[q1], core_node)
            movement_cost += weighted_distance_q1
        elif q1 not in current_assignment:
            movement_cost += 1
        if q2 in current_assignment and current_assignment[q2] != core_node:
            weighted_distance_q2 = get_dist(current_assignment[q2], core_node)
            movement_cost += weighted_distance_q2
        elif q2 not in current_assignment:
            movement_cost += 1

        attraction = 0.0
        max_lookahead = min(current_t + 4, len(timeslices))


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

        total_cost = movement_cost - attraction
        return max(0.1, total_cost)

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
        for i, timeslice in enumerate(timeslices):
            current_assignment = partition[i]
            for gate in timeslice:
                if len(gate.qubits) == 2 and gate.op.type not in [OpType.Measure, OpType.Reset]:
                    q1, q2 = gate.qubits[0], gate.qubits[1]
                    if current_assignment[q1] != current_assignment[q2]:
                        weighted_distance = self.get_weighted_distance(multicore_arch, 
                                                                      current_assignment[q1], 
                                                                      current_assignment[q2])
                        total_communication_cost += weighted_distance
        for i in range(2, len(timeslices)):
            prev_assignment = partition[i-1]
            curr_assignment = partition[i]
            for qubit in curr_assignment:
                if curr_assignment[qubit] != prev_assignment[qubit]:
                    weighted_distance = self.get_weighted_distance(multicore_arch,
                                                                  prev_assignment[qubit],
                                                                  curr_assignment[qubit])
                    total_communication_cost += weighted_distance
        return total_communication_cost
