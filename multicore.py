import random
import networkx as nx
import numpy as np
from pytket import OpType

def build_topology(topology, num_qpus_x, num_qpus_y=None):
    """
    Build and return a networkx.Graph for the requested topology.

    topology: one of ["line", "star", "tree", "mesh", "torus", "all_to_all"]
    num_qpus_x: primary size parameter
    num_qpus_y: secondary size parameter (required for mesh/torus)
    """
    topo = str(topology).lower()
    if topo == "line":
        G = nx.path_graph(num_qpus_x)

    elif topo == "star":
        total = num_qpus_x
        if total < 1:
            raise ValueError("star requires at least 1 node")
        G = nx.star_graph(total - 1)

    elif topo == "tree":
        G = nx.balanced_tree(r=2, h=num_qpus_x)

    elif topo == "mesh":
        if num_qpus_y is None:
            raise ValueError("mesh requires num_qpus_y")
        G = nx.grid_2d_graph(num_qpus_x, num_qpus_y)
        G = nx.convert_node_labels_to_integers(G)
        for i in range(num_qpus_x * num_qpus_y):
            x = i % num_qpus_x
            y = i // num_qpus_x
            G.nodes[i]['pos'] = (x, y)

    elif topo == "torus":
        if num_qpus_y is None:
            raise ValueError("torus requires num_qpus_y")
        G = nx.grid_2d_graph(num_qpus_x, num_qpus_y, periodic=True)
        G = nx.convert_node_labels_to_integers(G)
        for i in range(num_qpus_x * num_qpus_y):
            x = i % num_qpus_x
            y = i // num_qpus_x
            G.nodes[i]['pos'] = (x, y)

    elif topo == "all_to_all":
        total_nodes = num_qpus_x if num_qpus_y is None else num_qpus_x * num_qpus_y
        G = nx.complete_graph(total_nodes)
        for i in range(total_nodes):
            G.nodes[i]['pos'] = (i, 0)

    else:
        raise ValueError(f"Unknown topology: {topology}")

    return G

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
    def __init__(self, num_qpus_x, num_qpus_y, ncm_qubits=4, ncp_qubits=16, edge_weight_range=(1, 10), network_type='all_to_all'):
        print(f"Building a {network_type} network with {num_qpus_x} x {num_qpus_y} QPUs")
        self.network = build_topology(network_type, num_qpus_x, num_qpus_y)

        self._add_edge_weights(edge_weight_range)
        self.qpus = []
        self.collaboration_data = None
        
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
            weight = random.uniform(min_weight, max_weight)
            self.network[edge[0]][edge[1]]['weight'] = weight
    
    def _add_custom_edge_weights(self, weight_function=None):
        """
        Add custom weights to edges based on a function.
        
        Args:
            weight_function: Function that takes (node1, node2) and returns weight
        """
        if weight_function is None:
            def default_weight_func(n1, n2):
                p1 = self.network.nodes[n1].get('pos', None)
                p2 = self.network.nodes[n2].get('pos', None)
                if p1 is not None and p2 is not None:
                    manhattan_dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
                else:
                    manhattan_dist = abs(int(n1) - int(n2))
                base_weight = manhattan_dist + 1
                random_factor = random.uniform(0.5, 2.0)
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
                    try:
                        distance = nx.shortest_path_length(self.network, 
                                                         partition[qubits[0]], 
                                                         partition[qubits[1]], 
                                                         weight='weight')
                    except nx.NetworkXNoPath:
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
