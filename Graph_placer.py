import networkx as nx
from pytket import OpType

class graph_placer:
    def __init__(self, max_mappings=100, pruning_factor=0.5):
        """
        Initialize GraphPlacer with extensive search capabilities.
        
        Args:
            max_mappings: Maximum number of final mappings to keep (default: 100)
            pruning_factor: Factor for early pruning (0.5 means keep best 50% at each step)
        """
        self.max_mappings = max_mappings
        self.pruning_factor = pruning_factor
        self.all_mappings = []  # Store all final mappings with their costs
        
    def _get_weighted_distance(self, multicore_arch, node1, node2):
        """
        Get the weighted shortest path distance between two nodes.
        Falls back to unweighted distance if weights are not available.
        """
        try:
            distance = nx.shortest_path_length(multicore_arch.network, node1, node2, weight='weight')
            return distance
        except (nx.NetworkXNoPath, KeyError):
            try:
                distance = nx.shortest_path_length(multicore_arch.network, node1, node2)
                return distance
            except nx.NetworkXNoPath:
                return float('inf')

    def _calculate_partial_cost(self, partial_mapping, circuit, multicore_arch, timeslice_gates=None):
        """
        Calculate the communication cost for a partial mapping.
        Only considers gates where both qubits are already mapped.
        """
        cost = 0
        gates_to_check = timeslice_gates if timeslice_gates else list(circuit)
        
        for gate in gates_to_check:
            if len(gate.qubits) == 2:
                q1, q2 = gate.qubits[0], gate.qubits[1]
                if q1 in partial_mapping and q2 in partial_mapping:
                    if partial_mapping[q1] != partial_mapping[q2]:
                        distance = self._get_weighted_distance(
                            multicore_arch, partial_mapping[q1], partial_mapping[q2]
                        )
                        cost += distance
        return cost

    def _get_available_qpus(self, current_mapping, multicore_arch):
        """
        Get list of QPUs sorted by current utilization and their available capacity.
        """
        qpu_usage = {}
        qpu_nodes = list(multicore_arch.network.nodes())
        
        # Count current usage
        for qpu in qpu_nodes:
            qpu_usage[qpu] = 0
        
        for qpu in current_mapping.values():
            qpu_usage[qpu] = qpu_usage.get(qpu, 0) + 1
        
        # Return QPUs with available capacity, sorted by current usage (ascending)
        available_qpus = []
        for qpu in qpu_nodes:
            capacity = multicore_arch.network.nodes[qpu]['qpu'].available_qubits
            if qpu_usage[qpu] < capacity:
                available_qpus.append((qpu, qpu_usage[qpu], capacity - qpu_usage[qpu]))
        
        # Sort by usage (ascending) then by available capacity (descending)
        available_qpus.sort(key=lambda x: (x[1], -x[2]))
        return [qpu for qpu, _, _ in available_qpus]

    def _expand_mapping_options(self, current_mapping, next_qubit, multicore_arch, circuit, timeslice_gates=None):
        """
        Generate all possible placements for the next qubit and return them with costs.
        """
        options = []
        available_qpus = self._get_available_qpus(current_mapping, multicore_arch)
        
        for qpu in available_qpus:
            new_mapping = current_mapping.copy()
            new_mapping[next_qubit] = qpu
            
            # Calculate cost for this mapping option
            cost = self._calculate_partial_cost(new_mapping, circuit, multicore_arch, timeslice_gates)
            
            # Add heuristic penalty for load balancing
            qpu_load = sum(1 for q in new_mapping.values() if q == qpu)
            max_capacity = multicore_arch.network.nodes[qpu]['qpu'].available_qubits
            load_penalty = (qpu_load / max_capacity) * 0.1  # Small penalty for imbalance
            
            total_cost = cost + load_penalty
            options.append((new_mapping, total_cost))
        
        return options

    def _prune_mappings(self, mapping_options, keep_ratio=0.5):
        """
        Prune mapping options to keep only the most promising ones.
        """
        if keep_ratio is None:
            keep_ratio = self.pruning_factor
        
        if len(mapping_options) <= 10:  # Don't prune if we have few options
            return mapping_options
        
        # Sort by cost (ascending)
        mapping_options.sort(key=lambda x: x[1])
        
        # Keep the best percentage
        keep_count = max(5, int(len(mapping_options) * keep_ratio))
        keep_count = min(keep_count, len(mapping_options))
        
        return mapping_options[:keep_count]

    def _extensive_search_timeslice(self, timeslice_gates, multicore_arch):
        """
        Perform extensive search for a single timeslice with early pruning.
        Returns multiple mapping options with their costs.
        """
        # Get unique qubits in this timeslice
        qubits_in_slice = set()
        for gate in timeslice_gates:
            qubits_in_slice.update(gate.qubits)
        qubits_list = list(qubits_in_slice)
        
        if not qubits_list:
            return [({}, 0)]
        
        # Start with empty mapping
        current_level = [({}, 0)]  # (mapping, cost)
        
        # For each qubit, expand all current mappings
        for i, qubit in enumerate(qubits_list):
            next_level = []
            
            for current_mapping, current_cost in current_level:
                # Generate all placement options for this qubit
                options = self._expand_mapping_options(
                    current_mapping, qubit, multicore_arch, None, timeslice_gates
                )
                next_level.extend(options)
            
            # Prune to keep search tractable
            if len(next_level) > 200:  # Aggressive pruning during search
                prune_ratio = 0.3 if i < len(qubits_list) - 1 else 0.8  # Keep more at the end
                next_level = self._prune_mappings(next_level, prune_ratio)
            
            current_level = next_level
            
            print(f"  Qubit {i+1}/{len(qubits_list)}: {len(current_level)} mapping options")
        
        # Final pruning to get top mappings
        final_mappings = self._prune_mappings(current_level, 1.0)  # Keep all remaining
        return final_mappings[:self.max_mappings]  # Limit to max_mappings

    def break_into_timeslices(self, circuit):
        """
        Break circuit into timeslices while respecting dependencies.
        (Same as BFS placer implementation)
        """
        gates = list(circuit)
        num_gates = len(gates)

        # Build dependency graph
        predecessors = {i: set() for i in range(num_gates)}
        successors = {i: set() for i in range(num_gates)}

        last_used = {}
        for idx, gate in enumerate(gates):
            for q in gate.qubits:
                if q in last_used:
                    prev = last_used[q]
                    predecessors[idx].add(prev)
                    successors[prev].add(idx)
                last_used[q] = idx

        # Kahn's algorithm for layering
        ready = [i for i in range(num_gates) if not predecessors[i]]
        time_slices = []
        scheduled = set()

        while ready:
            current_slice = []
            used_qubits = set()
            next_ready = []

            for idx in ready:
                gate = gates[idx]
                gate_qubits = set(gate.qubits)
                if not used_qubits.intersection(gate_qubits):
                    current_slice.append(idx)
                    used_qubits.update(gate_qubits)
                    scheduled.add(idx)

            for idx in current_slice:
                for succ in successors[idx]:
                    if succ not in scheduled and predecessors[succ].issubset(scheduled):
                        next_ready.append(succ)

            time_slices.append([gates[i] for i in current_slice])
            ready = next_ready
            
        return time_slices

    def place_per_timeslice(self, circuit, multicore_arch):
        """
        Place circuit using extensive search with pruning for each timeslice.
        Returns the best mapping found.
        """
        timeslices = self.break_into_timeslices(circuit)
        best_overall_cost = float('inf')
        best_timeslice_partitions = []
        
        # For simplicity, we'll find the best mapping for each timeslice independently
        # In a more sophisticated version, we could consider cross-timeslice dependencies
        
        for i, timeslice in enumerate(timeslices):
            print(f"Processing timeslice {i+1}/{len(timeslices)}...")
            
            # Reset available qubits for this timeslice
            for node in multicore_arch.network.nodes():
                multicore_arch.network.nodes[node]['qpu'].available_qubits = multicore_arch.qpu_qubit_num
            
            # Get best mappings for this timeslice
            timeslice_mappings = self._extensive_search_timeslice(timeslice, multicore_arch)
            
            # For now, take the best mapping for this timeslice
            if timeslice_mappings:
                best_mapping = timeslice_mappings[0][0]  # Best mapping
                best_timeslice_partitions.append(best_mapping)
        
        return best_timeslice_partitions

    def get_all_mappings_with_costs(self, circuit, multicore_arch):
        """
        Get all discovered mappings with their costs for analysis.
        This is the main method that returns multiple mappings as requested.
        """
        timeslices = self.break_into_timeslices(circuit)
        all_mapping_combinations = []
        
        print(f"Analyzing circuit with {len(timeslices)} timeslices...")
        
        # Get all good mappings for each timeslice
        timeslice_options = []
        for i, timeslice in enumerate(timeslices):
            print(f"Generating options for timeslice {i+1}/{len(timeslices)}...")
            
            # Reset available qubits
            for node in multicore_arch.network.nodes():
                multicore_arch.network.nodes[node]['qpu'].available_qubits = multicore_arch.qpu_qubit_num
            
            mappings = self._extensive_search_timeslice(timeslice, multicore_arch)
            timeslice_options.append(mappings[:min(10, len(mappings))])  # Keep top 10 per timeslice
        
        # Generate combinations of timeslice mappings
        print("Generating mapping combinations...")
        from itertools import product
        
        # Limit combinations to avoid explosion
        max_combinations = min(self.max_mappings, 
                              min(100, len(timeslice_options[0]) * len(timeslice_options[-1])) if timeslice_options else 1)
        
        combination_count = 0
        for combination in product(*timeslice_options):
            if combination_count >= max_combinations:
                break
                
            # Extract the mappings and calculate total cost
            full_mapping = [mapping for mapping, cost in combination]
            total_cost = self.per_timeslice_cost(circuit, full_mapping, multicore_arch)
            
            all_mapping_combinations.append((full_mapping, total_cost))
            combination_count += 1
        
        # Sort by total cost and return top mappings
        all_mapping_combinations.sort(key=lambda x: x[1])
        final_mappings = all_mapping_combinations[:self.max_mappings]
        
        print(f"Generated {len(final_mappings)} final mapping options")
        return final_mappings

    def per_timeslice_cost(self, circuit, partition, multicore_arch):
        """
        Calculate per-timeslice cost using weighted distances.
        (Same as BFS placer implementation)
        """
        total_communication_cost = 0
        qubit_last_qpu = {}        
        timeslices = self.break_into_timeslices(circuit)
        
        for i in range(len(timeslices)):
            if i >= len(partition):
                break
                
            timeslice = timeslices[i]
            for gate in timeslice:
                gate_type = gate.op.type
                qubits = gate.qubits
                
                # Cost for two-qubit gates on different QPUs
                if len(qubits) == 2 and gate_type != OpType.Measure and gate_type != OpType.Reset:
                    if partition[i].get(qubits[0]) != partition[i].get(qubits[1]):
                        if qubits[0] in partition[i] and qubits[1] in partition[i]:
                            weighted_distance = self._get_weighted_distance(
                                multicore_arch, 
                                partition[i][qubits[0]], 
                                partition[i][qubits[1]]
                            )
                            total_communication_cost += weighted_distance

                # Cost for qubit movements between timeslices
                for qubit in qubits:
                    current_qpu = partition[i].get(qubit)
                    last_qpu = qubit_last_qpu.get(qubit)
                    
                    if last_qpu is not None and current_qpu != last_qpu:
                        weighted_distance = self._get_weighted_distance(
                            multicore_arch, last_qpu, current_qpu
                        )
                        total_communication_cost += weighted_distance
                        
                    if current_qpu is not None:
                        qubit_last_qpu[qubit] = current_qpu
                     
        return total_communication_cost

    def place(self, circuit, multicore_arch):
        """
        Standard place method for compatibility with main.py structure.
        Returns the best single mapping.
        """
        mappings_with_costs = self.get_all_mappings_with_costs(circuit, multicore_arch)
        if mappings_with_costs:
            # Return just the best mapping for compatibility
            best_mapping = mappings_with_costs[0][0]  # First element is the mapping
            # Convert to single partition format if needed
            if best_mapping and isinstance(best_mapping[0], dict):
                # Merge all timeslice mappings into one (taking the last occurrence)
                merged_mapping = {}
                for timeslice_mapping in best_mapping:
                    merged_mapping.update(timeslice_mapping)
                return merged_mapping
            return best_mapping
        return {}

    def get_mapping_statistics(self):
        """
        Get statistics about the search process.
        """
        if not hasattr(self, '_search_stats'):
            return {}
        
        return {
            'total_mappings_explored': self._search_stats.get('total_explored', 0),
            'final_mappings_kept': len(self.all_mappings),
            'pruning_efficiency': self._search_stats.get('pruning_efficiency', 0),
        }