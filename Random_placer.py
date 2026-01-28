from Placer import BasePlacer
import random
import numpy as np
import networkx as nx
from pytket import OpType
import matplotlib.pyplot as plt

class random_placer(BasePlacer):
    def place(self, circuit, multicore_arch):
        partition = {}
        qpu_nodes = list(multicore_arch.network.nodes())
        q_reg = circuit.qubits

        if len(q_reg) > len(qpu_nodes):
            print("Error: Not enough unique QPU nodes to place all qubits.")
            return None

        random_qpu_nodes = random.sample(qpu_nodes, len(q_reg))

        for qubit, qpu_node in zip(q_reg, random_qpu_nodes):
            partition[qubit] = qpu_node
        
        return partition

    def place_per_timeslice(self, circuit, multicore_arch):
        """
        Random placement per timeslice with weighted edge considerations.
        While placement is still random, cost calculations will use weighted distances.
        """
        timeslices = self.break_into_timeslices(circuit)
        timeslice_partitions = []

        for timeslice in timeslices:
            current_partition = {}

            # Reset QPU capacities for this timeslice
            for node in multicore_arch.network.nodes():
                multicore_arch.network.nodes[node]['qpu'].available_qubits = multicore_arch.qpu_qubit_num

            # Get all qubits in this timeslice
            qubits_in_timeslice = set()
            for gate in timeslice:
                qubits_in_timeslice.update(gate.qubits)
            
            qubits_in_timeslice = list(qubits_in_timeslice)

            # Random placement with capacity constraints
            for qubit in qubits_in_timeslice:
                if qubit not in current_partition:
                    available_qpus = [n for n in multicore_arch.network.nodes()
                                    if multicore_arch.network.nodes[n]['qpu'].available_qubits > 0]
                    if available_qpus:
                        chosen_qpu = random.choice(available_qpus)
                        current_partition[qubit] = chosen_qpu
                        multicore_arch.network.nodes[chosen_qpu]['qpu'].available_qubits -= 1

            timeslice_partitions.append(current_partition)

        return timeslice_partitions

    def place_per_timeslice_weighted_aware(self, circuit, multicore_arch, bias_factor=0.3):
        """
        Alternative random placement that slightly biases towards lower-weight edges.
        bias_factor: 0.0 = completely random, 1.0 = completely weight-based
        """
        timeslices = self.break_into_timeslices(circuit)
        timeslice_partitions = []

        for timeslice in timeslices:
            current_partition = {}

            # Reset QPU capacities for this timeslice
            for node in multicore_arch.network.nodes():
                multicore_arch.network.nodes[node]['qpu'].available_qubits = multicore_arch.qpu_qubit_num

            # First, place multi-qubit gates with slight bias towards lower weights
            placed_qubits = set()
            for gate in timeslice:
                if len(gate.qubits) == 2:
                    q1, q2 = gate.qubits
                    if q1 not in placed_qubits and q2 not in placed_qubits:
                        # Try to place both qubits on same QPU first
                        available_same_qpus = [n for n in multicore_arch.network.nodes()
                                             if multicore_arch.network.nodes[n]['qpu'].available_qubits >= 2]
                        
                        if available_same_qpus:
                            chosen_qpu = random.choice(available_same_qpus)
                            current_partition[q1] = chosen_qpu
                            current_partition[q2] = chosen_qpu
                            multicore_arch.network.nodes[chosen_qpu]['qpu'].available_qubits -= 2
                            placed_qubits.update([q1, q2])
                        else:
                            # Place on different QPUs, with slight bias towards lower-weight connections
                            available_qpus = [n for n in multicore_arch.network.nodes()
                                            if multicore_arch.network.nodes[n]['qpu'].available_qubits > 0]
                            
                            if len(available_qpus) >= 2:
                                # Create weighted selection based on edge weights
                                qpu_pairs = []
                                for i, qpu1 in enumerate(available_qpus):
                                    for j, qpu2 in enumerate(available_qpus[i+1:], i+1):
                                        if (multicore_arch.network.nodes[qpu1]['qpu'].available_qubits > 0 and
                                            multicore_arch.network.nodes[qpu2]['qpu'].available_qubits > 0):
                                            weight = self.get_weighted_distance(multicore_arch, qpu1, qpu2)
                                            # Lower weights get higher selection probability
                                            prob = 1.0 / (weight + 0.1) if weight != float('inf') else 0.1
                                            qpu_pairs.append(((qpu1, qpu2), prob))
                                
                                if qpu_pairs:
                                    # Combine random selection with weight bias
                                    if random.random() < bias_factor and qpu_pairs:
                                        # Weight-biased selection
                                        pairs, probs = zip(*qpu_pairs)
                                        total_prob = sum(probs)
                                        probs = [p/total_prob for p in probs]
                                        chosen_pair = np.random.choice(len(pairs), p=probs)
                                        qpu1, qpu2 = pairs[chosen_pair]
                                    else:
                                        # Random selection
                                        qpu1, qpu2 = random.choice([pair for pair, _ in qpu_pairs])
                                    
                                    current_partition[q1] = qpu1
                                    current_partition[q2] = qpu2
                                    multicore_arch.network.nodes[qpu1]['qpu'].available_qubits -= 1
                                    multicore_arch.network.nodes[qpu2]['qpu'].available_qubits -= 1
                                    placed_qubits.update([q1, q2])

            # Place remaining qubits randomly
            for gate in timeslice:
                for qubit in gate.qubits:
                    if qubit not in placed_qubits:
                        available_qpus = [n for n in multicore_arch.network.nodes()
                                        if multicore_arch.network.nodes[n]['qpu'].available_qubits > 0]
                        if available_qpus:
                            chosen_qpu = random.choice(available_qpus)
                            current_partition[qubit] = chosen_qpu
                            multicore_arch.network.nodes[chosen_qpu]['qpu'].available_qubits -= 1
                            placed_qubits.add(qubit)

            timeslice_partitions.append(current_partition)

        return timeslice_partitions

    def distribution_matrix(self, circuit, partition, multicore_arch):
        """
        Generate and save qubit distribution matrix visualization.
        """
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
        plt.title("Qubit Distribution Across Cores (Random Placer)")
        plt.xlabel("QPU X-coordinate")
        plt.ylabel("QPU Y-coordinate")
        plt.savefig("random_qubit_distribution_matrix.png", bbox_inches='tight')
        plt.close() 

    def per_timeslice_cost(self, circuit, partition, multicore_arch):
        """
        Calculate per-timeslice cost using weighted distances.
        """
        total_communication_cost = 0
        qubit_last_qpu = {}        
        timeslices = self.break_into_timeslices(circuit)
        
        for i in range(len(timeslices)):
            timeslice = timeslices[i]
            for gate in timeslice:
                type = gate.op.type
                qubits = gate.qubits
                
                # Cost for two-qubit gates on different QPUs (using weighted distance)
                if len(qubits) == 2 and type != OpType.Measure and type != OpType.Reset:
                    if partition[i][qubits[0]] != partition[i][qubits[1]]:
                        weighted_distance = self.get_weighted_distance(multicore_arch, 
                                                                      partition[i][qubits[0]], 
                                                                      partition[i][qubits[1]])
                        total_communication_cost += weighted_distance

                # Cost for qubit movements between timeslices (using weighted distance)
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

    def get_placement_statistics(self, circuit, partition, multicore_arch):
        """
        Get statistics about the random placement including weighted distances.
        """
        timeslices = self.break_into_timeslices(circuit)
        stats = {
            'total_timeslices': len(timeslices),
            'total_partitions': len(partition),
            'weighted_distances': [],
            'unweighted_distances': [],
            'two_qubit_gates': 0,
            'cross_qpu_gates': 0
        }
        
        for i in range(len(timeslices)):
            timeslice = timeslices[i]
            for gate in timeslice:
                if len(gate.qubits) == 2 and gate.op.type not in [OpType.Measure, OpType.Reset]:
                    stats['two_qubit_gates'] += 1
                    q1, q2 = gate.qubits
                    if partition[i][q1] != partition[i][q2]:
                        stats['cross_qpu_gates'] += 1
                        weighted_dist = self.get_weighted_distance(multicore_arch, 
                                                                  partition[i][q1], 
                                                                  partition[i][q2])
                        unweighted_dist = nx.shortest_path_length(multicore_arch.network, 
                                                                partition[i][q1], 
                                                                partition[i][q2])
                        stats['weighted_distances'].append(weighted_dist)
                        stats['unweighted_distances'].append(unweighted_dist)
        
        if stats['weighted_distances']:
            stats['avg_weighted_distance'] = np.mean(stats['weighted_distances'])
            stats['avg_unweighted_distance'] = np.mean(stats['unweighted_distances'])
        else:
            stats['avg_weighted_distance'] = 0
            stats['avg_unweighted_distance'] = 0
            
        return stats

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
            partition = self.place_per_timeslice_weighted_aware(circuit, multicore_arch)
            cost = self.per_timeslice_cost(circuit, partition, multicore_arch)

            # Store the result
            results.append((broken_links, cost))

        # Restore the original network
        multicore_arch.network = original_network

        print(results)
        return results