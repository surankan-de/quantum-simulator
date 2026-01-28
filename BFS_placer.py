import random
from collections import deque
from pytket import OpType
from Placer import BasePlacer

class bfs_placer(BasePlacer):
    def _get_weighted_neighbors_sorted(self, multicore_arch, node):
        neighbors = list(multicore_arch.network.neighbors(node))
        if not neighbors:
            return neighbors
        
        neighbor_distances = []
        for neighbor in neighbors:
            distance = self.get_weighted_distance(multicore_arch, node, neighbor)
            neighbor_distances.append((neighbor, distance))
        
        neighbor_distances.sort(key=lambda x: x[1])
        return [neighbor for neighbor, _ in neighbor_distances]

    def place(self, circuit, multicore_arch):
        partition = {}
        qpu_nodes = list(multicore_arch.network.nodes())
        q_reg = list(circuit.qubits)
        
        two_qubit_gates = [gate for gate in circuit if len(gate.qubits) == 2]
        placed_qubits = set()
        
        for gate in two_qubit_gates:
            qubit1, qubit2 = gate.qubits[0], gate.qubits[1]

            if qubit1 in placed_qubits or qubit2 in placed_qubits:
                continue

            for qpu in qpu_nodes:
                if multicore_arch.network.nodes[qpu]['qpu'].available_qubits >= 2:
                    partition[qubit1] = qpu
                    partition[qubit2] = qpu
                    multicore_arch.network.nodes[qpu]['qpu'].available_qubits -= 2
                    placed_qubits.add(qubit1)
                    placed_qubits.add(qubit2)
                    break
            
            if qubit1 not in placed_qubits:
                found_adjacent = False
                random.shuffle(qpu_nodes)
                for qpu1 in qpu_nodes:
                    if multicore_arch.network.nodes[qpu1]['qpu'].available_qubits >= 1:
                        sorted_neighbors = self._get_weighted_neighbors_sorted(multicore_arch, qpu1)
                        for qpu2 in sorted_neighbors:
                            if multicore_arch.network.nodes[qpu2]['qpu'].available_qubits >= 1:
                                partition[qubit1] = qpu1
                                partition[qubit2] = qpu2
                                multicore_arch.network.nodes[qpu1]['qpu'].available_qubits -= 1
                                multicore_arch.network.nodes[qpu2]['qpu'].available_qubits -= 1
                                placed_qubits.add(qubit1)
                                placed_qubits.add(qubit2)
                                found_adjacent = True
                                break
                        if found_adjacent:
                            break
        
        unplaced_qubits = [q for q in q_reg if q not in placed_qubits]
        
        if not unplaced_qubits:
            return partition

        available_qpus = [(qpu, multicore_arch.network.nodes[qpu]['qpu'].available_qubits) 
                         for qpu in qpu_nodes 
                         if multicore_arch.network.nodes[qpu]['qpu'].available_qubits > 0]
        
        available_qpus.sort(key=lambda x: x[1], reverse=True)
        
        queue = deque()
        visited = set()
        for qpu, _ in available_qpus:
            queue.append(qpu)
            visited.add(qpu)

        while queue and unplaced_qubits:
            current_qpu = queue.popleft()

            qubit_to_place = unplaced_qubits.pop(0)
            partition[qubit_to_place] = current_qpu
            multicore_arch.network.nodes[current_qpu]['qpu'].available_qubits -= 1

            sorted_neighbors = self._get_weighted_neighbors_sorted(multicore_arch, current_qpu)
            for neighbor_qpu in sorted_neighbors:
                if neighbor_qpu not in visited and multicore_arch.network.nodes[neighbor_qpu]['qpu'].available_qubits > 0:
                    visited.add(neighbor_qpu)
                    queue.append(neighbor_qpu)

        for qubit in unplaced_qubits:
            available_qpus = [q for q in qpu_nodes if multicore_arch.network.nodes[q]['qpu'].available_qubits > 0]
            if available_qpus:
                chosen_qpu = random.choice(available_qpus)
                partition[qubit] = chosen_qpu
                multicore_arch.network.nodes[chosen_qpu]['qpu'].available_qubits -= 1

        return partition

    def place_per_timeslice(self, circuit, multicore_arch):
        timeslices = self.break_into_timeslices(circuit)
        timeslice_partitions = []

        for timeslice in timeslices:
            for node in multicore_arch.network.nodes():
                multicore_arch.network.nodes[node]['qpu'].available_qubits = multicore_arch.qpu_qubit_num

            current_partition = {}
            qpu_nodes = list(multicore_arch.network.nodes())
            
            qubits_in_slice = set()
            for gate in timeslice:
                qubits_in_slice.update(gate.qubits)
            qubits_in_slice = list(qubits_in_slice)
            
            temp_available_qubits = {qpu: multicore_arch.network.nodes[qpu]['qpu'].available_qubits for qpu in qpu_nodes}
            
            placed_qubits_in_slice = set()
            for gate in timeslice:
                if len(gate.qubits) == 2:
                    q1, q2 = gate.qubits[0], gate.qubits[1]
                    if q1 in placed_qubits_in_slice or q2 in placed_qubits_in_slice:
                        continue
                    
                    found_placement = False
                    
                    for qpu in qpu_nodes:
                        if temp_available_qubits.get(qpu, 0) >= 2:
                            current_partition[q1] = qpu
                            current_partition[q2] = qpu
                            temp_available_qubits[qpu] -= 2
                            placed_qubits_in_slice.add(q1)
                            placed_qubits_in_slice.add(q2)
                            found_placement = True
                            break
                    
                    if not found_placement:
                        qpu_pairs = []
                        for qpu1 in qpu_nodes:
                            if temp_available_qubits.get(qpu1, 0) >= 1:
                                sorted_neighbors = self._get_weighted_neighbors_sorted(multicore_arch, qpu1)
                                for qpu2 in sorted_neighbors:
                                    if temp_available_qubits.get(qpu2, 0) >= 1:
                                        distance = self.get_weighted_distance(multicore_arch, qpu1, qpu2)
                                        qpu_pairs.append((qpu1, qpu2, distance))
                        
                        qpu_pairs.sort(key=lambda x: x[2])
                        
                        for qpu1, qpu2, distance in qpu_pairs:
                            if (temp_available_qubits.get(qpu1, 0) >= 1 and 
                                temp_available_qubits.get(qpu2, 0) >= 1):
                                current_partition[q1] = qpu1
                                current_partition[q2] = qpu2
                                temp_available_qubits[qpu1] -= 1
                                temp_available_qubits[qpu2] -= 1
                                placed_qubits_in_slice.add(q1)
                                placed_qubits_in_slice.add(q2)
                                found_placement = True
                                break

                if len(gate.qubits) == 1:
                    q = gate.qubits[0]
                    if q not in placed_qubits_in_slice:
                        available_qpus = [qpu for qpu in qpu_nodes if temp_available_qubits.get(qpu, 0) > 0]
                        if available_qpus:
                            chosen_qpu = random.choice(available_qpus)
                            current_partition[q] = chosen_qpu
                            temp_available_qubits[chosen_qpu] -= 1
                            placed_qubits_in_slice.add(q)
            
            unplaced_qubits = [q for q in qubits_in_slice if q not in placed_qubits_in_slice]
            
            available_qpu_list = [(qpu, temp_available_qubits.get(qpu, 0)) 
                                 for qpu in qpu_nodes 
                                 if temp_available_qubits.get(qpu, 0) > 0]
            available_qpu_list.sort(key=lambda x: x[1], reverse=True)
            
            queue = deque()
            visited = set()
            for qpu, _ in available_qpu_list:
                queue.append(qpu)
                visited.add(qpu)
            
            while queue and unplaced_qubits:
                current_qpu = queue.popleft()
                
                qubit_to_place = unplaced_qubits.pop(0)
                current_partition[qubit_to_place] = current_qpu
                temp_available_qubits[current_qpu] -= 1
                
                sorted_neighbors = self._get_weighted_neighbors_sorted(multicore_arch, current_qpu)
                for neighbor_qpu in sorted_neighbors:
                    if neighbor_qpu not in visited and temp_available_qubits.get(neighbor_qpu, 0) > 0:
                        visited.add(neighbor_qpu)
                        queue.append(neighbor_qpu)
            
            timeslice_partitions.append(current_partition)
        
        return timeslice_partitions

    def per_timeslice_cost(self, circuit, partition, multicore_arch):
        """
        calculates the total weighted intercore communications
        
        :param circuit: the logical circuit
        :param partition: the mapping generated by the placer
        :param multicore_arch: the physical architecture
        """
        total_communication_cost = 0
        qubit_last_qpu = {}        
        timeslices = self.break_into_timeslices(circuit)
        
        for i in range(len(timeslices)):
            timeslice = timeslices[i]
            for gate in timeslice:
                type = gate.op.type
                qubits = gate.qubits
                
                if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                    if partition[i][qubits[0]] != partition[i][qubits[1]]:
                        weighted_distance = self.get_weighted_distance(multicore_arch, 
                                                                      partition[i][qubits[0]], 
                                                                      partition[i][qubits[1]])
                        total_communication_cost += weighted_distance

                for qubit in qubits:
                    current_qpu = partition[i].get(qubit)
                    last_qpu = qubit_last_qpu.get(qubit)
                    
                    if last_qpu is not None and current_qpu != last_qpu:
                        weighted_distance = self.get_weighted_distance(multicore_arch, 
                                                                      last_qpu, 
                                                                      current_qpu)
                        total_communication_cost += weighted_distance
                        
                    qubit_last_qpu[qubit] = current_qpu
                     
        return total_communication_cost

    def simulate_broken_links(self, circuit, multicore_arch, num_simulations=10, break_probability=0.1):
        """
        Simulate mappings with subsets of links broken (infinite weight) and calculate costs.
        
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
            simulated_network = original_network.copy()
            broken_links = set()
            for edge in original_network.edges:
                if random.random() < break_probability:
                    simulated_network[edge[0]][edge[1]]['weight'] = float('inf')
                    broken_links.add(edge)

            multicore_arch.network = simulated_network

            partition = self.place_per_timeslice(circuit, multicore_arch)
            cost = self.per_timeslice_cost(circuit, partition, multicore_arch)

            # Store the result
            results.append((broken_links, cost))

        multicore_arch.network = original_network

        print(results)
        return results