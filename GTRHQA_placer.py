from Placer import BasePlacer
from pytket import OpType
import random

def random_initial_assignment(all_qubits, core_nodes, qubits_per_core, seed=None):
    if seed is not None:
        random.seed(seed)

    # create a multiset of cores with exact capacity
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

        # ---------- Initial round-robin assignment ----------
        initial_assignment = {q: core_nodes[int(i / qubits_per_core)] for i, q in enumerate(all_qubits)}
        # print("Initial assignment:", initial_assignment)
        # initial_assignment = random_initial_assignment(all_qubits,core_nodes,qubits_per_core)

        # partition.append(initial_assignment)

        # ---------- Precompute weighted distance matrix ----------
        dist_matrix = np.zeros((num_cores, num_cores))
        self._dist_matrix = dist_matrix
        self.node_index = {node: i for i, node in enumerate(core_nodes)}
        for i, n1 in enumerate(core_nodes):
            for j, n2 in enumerate(core_nodes):
                if i != j:
                    dist_matrix[i, j] = self.get_weighted_distance(multicore_arch, n1, n2)
        dist_matrix[np.isnan(dist_matrix)] = np.inf  
             


        # ---------- Process each timeslice ----------
        for t in range(0, len(timeslices)):
            current_timeslice = timeslices[t]
            if(t>0):

                prev_assignment = partition[t - 1]
            else:
                prev_assignment = initial_assignment
            new_assignment = dict(prev_assignment)  # shallow copy (not deepcopy)

            # Identify 2-qubit gates that span different cores
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

            # Track core occupancy
            core_occupancy = {node: 0 for node in core_nodes}
            for q, c in new_assignment.items():
                core_occupancy[c] += 1

            available_cores = [node for node in core_nodes if core_occupancy[node] <= qubits_per_core - 2]
            if len(available_cores) == 0:
                partition.append(new_assignment)
                continue

            num_ops = len(unfeasible_ops)
            num_avail = len(available_cores)

            # ---------- Vectorized cost matrix computation ----------
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

            # ---------- Greedy vectorized assignment ----------
            
            # ==== Fast vectorized greedy assignment (replacement for argsort loop) ====
            # Convert core occupancy and available_cores into index-based NumPy structures
            avail_count = len(available_cores)
            num_ops = len(unfeasible_ops)

            # Map available_cores -> core indices in core_nodes
            avail_core_idxs = np.array([self.node_index[c] for c in available_cores], dtype=np.int32)
            # occupancy array over available cores (floats ok for occupancy penalty)
            occupancy_arr = np.array([core_occupancy[c] for c in available_cores], dtype=np.float32)

            # Guard: nothing to do
            if num_ops == 0 or avail_count == 0:
                # nothing assignable
                pass
            else:
                # We will keep track of assigned operations and which cores remain available
                assigned_op_for_core = -np.ones(avail_count, dtype=np.int32)  # for each available core: op index assigned or -1
                assigned_core_for_op = -np.ones(num_ops, dtype=np.int32)      # for each op: core idx in available_cores or -1
                core_available_mask = np.ones(avail_count, dtype=bool)

                # Precompute a boolean mask of infinite costs (disallowed assignments)
                finite_mask = np.isfinite(cost_matrix)

                # Iteratively assign in rounds:
                # Round: each unassigned op picks its best available core (argmin over available columns).
                # Then for each core chosen by multiple ops, pick the op with smallest cost for that core.
                # Mark chosen cores used; remove assigned ops; repeat until no progress or no cores left.
                ops_remaining_mask = np.ones(num_ops, dtype=bool)
                # to avoid pathological infinite loops cap iterations
                max_rounds = min(avail_count, num_ops) + 2
                round_i = 0

                while ops_remaining_mask.any() and core_available_mask.any() and round_i < max_rounds:
                    round_i += 1

                    # Select columns (available cores)
                    cols_idx = np.nonzero(core_available_mask)[0]
                    if cols_idx.size == 0:
                        break

                    # Submatrix of costs for remaining ops vs available cores
                    rows_idx = np.nonzero(ops_remaining_mask)[0]
                    sub = cost_matrix[rows_idx][:, cols_idx]  # shape (n_rem_ops, n_avail_cols)

                    # If sub has no finite entries, break
                    if not np.isfinite(sub).any():
                        break

                    # For each remaining op, pick its best column among available ones
                    # `rel_col` is relative column index in cols_idx
                    # Use argmin but mask infinities: set large sentinel where not finite
                    huge = 1e12
                    sub_masked = np.where(np.isfinite(sub), sub, huge)
                    rel_col = np.argmin(sub_masked, axis=1)                   # chosen relative column for each remaining op
                    chosen_costs = sub_masked[np.arange(sub_masked.shape[0]), rel_col]
                    chosen_cores = cols_idx[rel_col]                          # map to absolute available_cores indices
                    chosen_ops = rows_idx                                     # rows_idx[i] corresponds to sub row i

                    # Resolve conflicts: for each chosen core, pick the single op with minimal cost
                    # We'll group by chosen_cores
                    unique_cores, inv = np.unique(chosen_cores, return_inverse=True)
                    picks_rows = []
                    picks_cores = []
                    for ui, core_idx in enumerate(unique_cores):
                        idxs = np.where(inv == ui)[0]  # indices into chosen_ops/chosen_costs
                        if idxs.size == 0:
                            continue
                        # select the index among idxs with min cost
                        local_costs = chosen_costs[idxs]
                        best_local_i = idxs[int(np.argmin(local_costs))]
                        op_to_assign = int(chosen_ops[best_local_i])  # op index in original unfeasible_ops
                        picks_rows.append(op_to_assign)
                        picks_cores.append(int(core_idx))

                    # Apply picks: assign these ops to these cores
                    for op_idx, core_idx in zip(picks_rows, picks_cores):
                        # only assign if still unassigned and core still available and cost finite
                        if assigned_core_for_op[op_idx] != -1:
                            continue
                        if not core_available_mask[core_idx]:
                            continue
                        cost_val = cost_matrix[op_idx, core_idx]
                        if not np.isfinite(cost_val):
                            continue

                        # mark assignment
                        assigned_core_for_op[op_idx] = core_idx
                        assigned_op_for_core[core_idx] = op_idx
                        core_available_mask[core_idx] = False
                        ops_remaining_mask[op_idx] = False

                        # update occupancy and new_assignment for the two qubits of this op
                        gate, q1, q2 = unfeasible_ops[op_idx]
                        core_node = available_cores[core_idx]

                        # decrement occupancy of previous core for q1/q2 if present
                        if q1 in new_assignment:
                            core_occupancy[new_assignment[q1]] -= 1
                        if q2 in new_assignment:
                            core_occupancy[new_assignment[q2]] -= 1

                        new_assignment[q1] = core_node
                        new_assignment[q2] = core_node

                        # update local occupancy_arr (keep it consistent)
                        occupancy_arr[core_idx] += 2

                    # end of round: continue to next round to assign remaining ops to remaining cores

                # After iterative rounds, assign any still-unassigned ops greedily to any remaining cores
                remaining_ops = np.where(assigned_core_for_op == -1)[0]
                remaining_core_idxs = np.where(core_available_mask)[0].tolist()
                if remaining_ops.size > 0 and len(remaining_core_idxs) > 0:
                    for op_idx in remaining_ops:
                        # if no cores left break
                        if not remaining_core_idxs:
                            break
                        # pick the core with minimum cost for this op among remaining cores
                        costs_for_op = cost_matrix[op_idx, remaining_core_idxs]
                        # mask infinities
                        finite_mask_local = np.isfinite(costs_for_op)
                        if not finite_mask_local.any():
                            continue
                        rel = int(np.argmin(np.where(finite_mask_local, costs_for_op, 1e12)))
                        chosen_core_rel = remaining_core_idxs[rel]
                        # assign
                        gate, q1, q2 = unfeasible_ops[op_idx]
                        core_node = available_cores[chosen_core_rel]

                        if q1 in new_assignment:
                            core_occupancy[new_assignment[q1]] -= 1
                        if q2 in new_assignment:
                            core_occupancy[new_assignment[q2]] -= 1

                        new_assignment[q1] = core_node
                        new_assignment[q2] = core_node

                        occupancy_arr[chosen_core_rel] += 2
                        # mark core used and remove from list
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
        # for t in range(current_t + 1, max_lookahead):
        #     weight = 2 ** -(t - current_t)
        #     timeslice = timeslices[t]
        #     for gate in timeslice:
        #         if len(gate.qubits) == 2:
        #             gate_q1, gate_q2 = gate.qubits[0], gate.qubits[1]
        #             for assigned_qubit, assigned_core in current_assignment.items():
        #                 if assigned_core == core_node:
        #                     if ((gate_q1 == q1 and gate_q2 == assigned_qubit) or
        #                         (gate_q2 == q1 and gate_q1 == assigned_qubit) or
        #                         (gate_q1 == q2 and gate_q2 == assigned_qubit) or
        #                         (gate_q2 == q2 and gate_q1 == assigned_qubit)):
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
