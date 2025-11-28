import networkx as nx
import numpy as np
from collections import deque
from pytket import Circuit, OpType
from scipy.optimize import linear_sum_assignment

class BasePlacer:
    """
    Base class for quantum circuit placers with common functionality.
    Different placement strategies inherit from this class.
    """
    
    def __init__(self):
        """Initialize common parameters used by all placers."""
        self.name = "BasePlacer"
    
    def break_into_timeslices(self, circuit):
        """
        Break circuit into timeslices while respecting dependencies.
        Common to all placers.
        """
        gates = list(circuit)
        num_gates = len(gates)
        print(len(gates),"gatelen")
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
        print(sum(len(i) for i in time_slices),"got")   
        return time_slices
    
    def get_weighted_distance(self, multicore_arch, node1, node2):
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
    
    def per_timeslice_cost(self, circuit, partition, multicore_arch):
        """
        Calculate per-timeslice cost using weighted distances.
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
                            weighted_distance = self.get_weighted_distance(
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
                        weighted_distance = self.get_weighted_distance(
                            multicore_arch, last_qpu, current_qpu
                        )
                        total_communication_cost += weighted_distance
                        
                    if current_qpu is not None:
                        qubit_last_qpu[qubit] = current_qpu
                     
        return total_communication_cost
    
    def place_per_timeslice(self, circuit, multicore_arch):
        """
        Base implementation of place_per_timeslice.
        Should be overridden by subclasses.
        """
        raise NotImplementedError("Subclasses must implement place_per_timeslice method")
    
    # Other common utility methods
    def get_qpu_capacity(self, qpu_node, multicore_arch):
        """Get the capacity of a QPU."""
        return multicore_arch.network.nodes[qpu_node]['qpu'].available_qubits

