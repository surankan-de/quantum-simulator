"""
des_simulator.py

Discrete Event Simulator with queued interconnects.

Key behavior:
- Edges have FIFO queues. If an edge is busy, messages wait in queue.
- Hop consumes hop_latency while occupying the edge.
- Edge success prob = 1 / edge['weight'].
- On hop failure, retry the same hop from the same node after backoff.
- Timeslices are scheduled at fixed times; transfers may run across boundaries.
- Tracks detailed stats: transfers, hops, queuing wait, inflight at boundaries, crossing boundaries, total simulated time.

Inputs:
- multicore_arch: object with .network (networkx Graph) where edges have 'weight'
- partition: list of dicts per timeslice mapping logical_qubit -> node
- timeslices: list of timeslices; each timeslice is list of gate-like objects with `.qubits`
"""

import heapq
import random
from collections import defaultdict, deque, namedtuple
import networkx as nx
import math
from typing import Tuple

Event = namedtuple("Event", ["time", "seq", "etype", "payload"])

class DES_Simulator:
    def __init__(
        self,
        multicore_arch,
        timeslices,
        partition,
        seed=None,
        hop_latency=1,
        gate_latency=1,
        max_retries=50,
        backoff_fn=None,
        timeslice_interval=1.0,
        verbose=False,
    ):
        self.multicore = multicore_arch
        self.net = getattr(multicore_arch, "network", None)
        if self.net is None or not isinstance(self.net, (nx.Graph, nx.DiGraph)):
            raise ValueError("multicore_arch must have a .network (networkx Graph/DiGraph)")

        self.partition = partition
        self.timeslices = timeslices
        self.hop_latency = float(hop_latency)
        self.gate_latency = float(gate_latency)
        self.max_retries = int(max_retries)
        self.backoff_fn = backoff_fn or (lambda r: 1)  # exponential default
        self.timeslice_interval = float(timeslice_interval)
        self.verbose = verbose

        self.rng = random.Random(seed)

        # event queue
        self._evq = []
        self._seq = 0

        # inflight transfers: msg_id -> meta
        self._inflight = {}
        self._next_msg_id = 0

        # per-edge queues and busy state
        # edges keyed as (u,v) with u->v direction matching path order
        self.edge_queues = defaultdict(deque)      # (u,v) -> deque of msg_id
        self.edge_busy_until = defaultdict(float)  # (u,v) -> time when edge becomes free
        self.edge_busy_count = defaultdict(int)    # counts how many hops used this edge (utilization metric)

        # completed transfer metas (store for stats like crossing boundaries)
        self._completed_metas = []

        # stats
        self.stats = {
            "total_two_qubit_gates": 0,           # all two-qubit gates observed
            "total_intercore_transfers": 0,       # messages created for inter-core gates
            "completed_transfers": 0,             # transfers reached destination
            "successful_two_qubit_gates": 0,      # completed remote gates
            "failed_two_qubit_gates": 0,          # transfers failed permanently
            "transfer_attempts": 0,               # hop attempts (including retries)
            "successful_hops": 0,
            "failed_hops": 0,
            "per_core_comm": defaultdict(int),
            "gate_latencies": [],                 # end-to-end times (start->gate_done)
            "inflight_at_boundary": [],           # number of inflight transfers at each timeslice boundary
            "transfers_crossing_boundary": 0,
            "total_queue_wait": 0.0,              # sum of queue wait times (for avg)
            "queue_wait_events": 0,
            "total_sim_time": 0.0,
            "per_edge_queue_lengths_max": defaultdict(int),
            "per_edge_total_queue_time": defaultdict(float),
        }

    # ----------------------
    # Event queue helpers
    # ----------------------
    def _push_event(self, time_, etype, payload):
        heapq.heappush(self._evq, Event(time_, self._seq, etype, payload))
        self._seq += 1

    def _pop_event(self):
        return heapq.heappop(self._evq) if self._evq else None

    # ----------------------
    # Network helpers
    # ----------------------
    def _edge_key(self, u, v) -> Tuple:
        # directed key used in path order
        return (u, v)

    def _edge_prob(self, u, v):
        w = self.net[u][v].get("weight", 1.0)
        if w <= 0:
            return 0.0
        return min(max(1.0 / float(w), 0.0), 1.0)

    def _shortest_path_nodes(self, src, dst):
        try:
            return nx.shortest_path(self.net, src, dst)
        except nx.NetworkXNoPath:
            return None

    # ----------------------
    # Core transfer lifecycle
    # ----------------------
    def _create_transfer(self, gate, slice_start, node_src, node_dst, slice_idx):
        """Create transfer meta and schedule the first hop start event at slice_start."""
        path_nodes = self._shortest_path_nodes(node_src, node_dst)
        if not path_nodes:
            if self.verbose:
                print(f"[WARN] No path {node_src}->{node_dst} for gate; marking failed.")
            self.stats["failed_two_qubit_gates"] += 1
            return None

        msg_id = self._next_msg_id
        self._next_msg_id += 1
        meta = {
            "msg_id": msg_id,
            "gate": gate,
            "slice_idx": slice_idx,
            "src": node_src,
            "dst": node_dst,
            "path_nodes": path_nodes,
            "next_hop_idx": 0,        # index of current node (will attempt hop to next)
            "hop_retries": 0,         # retries for current hop
            "start_time": slice_start,
            "end_time": None,
            "queued_since": None,     # when it entered a specific edge queue (for wait accounting)
            "crossed_boundaries": 0,
        }
        self._inflight[msg_id] = meta
        # schedule first hop start
        self._push_event(slice_start, "start_hop", {"msg_id": msg_id})
        return meta

    # ----------------------
    # Event Handlers
    # ----------------------
    def _handle_execute_local_gate(self, ev):
        gate = ev.payload["gate"]
        slice_start = ev.payload["slice_start"]
        finish = ev.time + self.gate_latency
        if self.verbose:
            print(f"[t={ev.time}] Local gate {getattr(gate,'name',repr(gate))} -> done @ {finish}")
        self._push_event(finish, "local_gate_done", {"slice_start": slice_start})

    def _handle_local_gate_done(self, ev):
        self.stats["gate_latencies"].append(ev.time - ev.payload["slice_start"])

    def _handle_start_hop(self, ev):
        msg_id = ev.payload["msg_id"]
        meta = self._inflight.get(msg_id)
        if meta is None:
            return  # gone
        idx = meta["next_hop_idx"]
        path = meta["path_nodes"]
        if idx >= len(path) - 1:
            # already at destination
            self._push_event(ev.time, "transfer_arrived", {"msg_id": msg_id})
            return

        u = path[idx]
        v = path[idx + 1]
        edge = self._edge_key(u, v)
        now = ev.time

        # if edge is free at now -> occupy immediately
        busy_until = self.edge_busy_until.get(edge, 0.0)
        if busy_until <= now and len(self.edge_queues[edge]) == 0:
            # start traversal immediately
            # account for any queue wait (none)
            self.edge_busy_until[edge] = now + self.hop_latency
            self.edge_busy_count[edge] += 1
            # schedule finish_hop at now + hop_latency
            if self.verbose:
                print(f"[t={now}] msg#{msg_id} starting hop {u}->{v} (edge free) -> finishes @ {now + self.hop_latency}")
            # update per-edge max queue length
            self.stats["per_edge_queue_lengths_max"][edge] = max(self.stats["per_edge_queue_lengths_max"].get(edge,0), 0)
            self._push_event(now + self.hop_latency, "finish_hop", {"msg_id": msg_id, "edge": edge, "u": u, "v": v})
        else:
            # enqueue msg_id on edge queue
            q = self.edge_queues[edge]
            q.append(msg_id)
            # record queue length stat
            self.stats["per_edge_queue_lengths_max"][edge] = max(self.stats["per_edge_queue_lengths_max"].get(edge,0), len(q))
            # mark when it entered queue
            meta["queued_since"] = now
            if self.verbose:
                print(f"[t={now}] msg#{msg_id} queued on edge {u}->{v} pos={len(q)} (busy_until={busy_until})")

    def _start_next_on_edge(self, edge, now):
        """Pop next message from edge queue and start its hop immediately at time 'now'."""
        q = self.edge_queues[edge]
        if not q:
            return
        next_msg = q.popleft()
        # compute wait time for this msg
        meta = self._inflight.get(next_msg)
        if meta is None:
            # message disappeared, skip
            return
        if meta.get("queued_since") is not None:
            wait = now - meta["queued_since"]
            self.stats["total_queue_wait"] += wait
            self.stats["queue_wait_events"] += 1
            # accumulate per-edge queue time
            self.stats["per_edge_total_queue_time"][edge] += wait
            meta["queued_since"] = None

        # occupy edge now
        self.edge_busy_until[edge] = now + self.hop_latency
        self.edge_busy_count[edge] += 1
        u, v = edge
        if self.verbose:
            print(f"[t={now}] dequeued msg#{next_msg} on edge {u}->{v} -> finishes @ {now + self.hop_latency}")
        self._push_event(now + self.hop_latency, "finish_hop", {"msg_id": next_msg, "edge": edge, "u": u, "v": v})

    def _handle_finish_hop(self, ev):
        msg_id = ev.payload["msg_id"]
        edge = ev.payload["edge"]
        u = ev.payload["u"]
        v = ev.payload["v"]
        now = ev.time

        meta = self._inflight.get(msg_id)
        if meta is None:
            # might have been removed by failure handling; still free edge and start next
            # free edge
            if self.edge_busy_until.get(edge, 0.0) <= now:
                # start next queued
                self._start_next_on_edge(edge, now)
            return

        # a hop attempt just finished (we consumed hop_latency)
        self.stats["transfer_attempts"] += 1

        # sample hop success
        p_succ = self._edge_prob(u, v)
        if self.rng.random() < p_succ:
            # success
            self.stats["successful_hops"] += 1
            if self.verbose:
                print(f"[t={now}] msg#{msg_id} hop {u}->{v} SUCCESS")
            # advance to next hop
            meta["next_hop_idx"] += 1
            meta["hop_retries"] = 0
            # update per-core comm counters
            self.stats["per_core_comm"][u] += 1
            self.stats["per_core_comm"][v] += 1
            # if reached destination
            if meta["next_hop_idx"] >= len(meta["path_nodes"]) - 1:
                # schedule arrival -> remote gate execution
                self._push_event(now, "transfer_arrived", {"msg_id": msg_id})
            else:
                # schedule next hop start at now (will check edge queue state)
                self._push_event(now, "start_hop", {"msg_id": msg_id})
        else:
            # failure
            self.stats["failed_hops"] += 1
            meta["hop_retries"] = meta.get("hop_retries", 0) + 1
            if self.verbose:
                print(f"[t={now}] msg#{msg_id} hop {u}->{v} FAILED (retry #{meta['hop_retries']})")
            if meta["hop_retries"] > self.max_retries:
                # permanent fail: mark failed and remove inflight
                meta["end_time"] = now
                self._inflight.pop(msg_id, None)
                self.stats["failed_two_qubit_gates"] += 1
                if self.verbose:
                    print(f"[t={now}] msg#{msg_id} TRANSFER FAILED permanently")
            else:
                # schedule retry start after backoff (retry will attempt to occupy edge, queue if busy)
                delay = self.backoff_fn(meta["hop_retries"])
                retry_time = now + delay
                if self.verbose:
                    print(f"[t={now}] msg#{msg_id} will retry hop {u}->{v} at t={retry_time}")
                self._push_event(retry_time, "start_hop", {"msg_id": msg_id})

        # free edge now and start next queued on this edge (if any)
        # Note: edge is free at ev.time since finish_hop has occurred
        # but other events may schedule immediate usage; we pop next and start at 'now'
        # set busy_until only if we start next
        # start next if queue not empty
        self.edge_busy_until[edge] = now  # free now; will be set by _start_next_on_edge
        self._start_next_on_edge(edge, now)

    def _handle_transfer_arrived(self, ev):
        msg_id = ev.payload["msg_id"]
        meta = self._inflight.get(msg_id)
        if meta is None:
            return
        now = ev.time
        meta["end_time"] = now
        # schedule remote gate execution finishing after gate_latency
        finish = now + self.gate_latency
        if self.verbose:
            print(f"[t={now}] msg#{msg_id} reached dst {meta['dst']}; scheduling remote gate done @ {finish}")
        self._push_event(finish, "remote_gate_done", {"msg_id": msg_id})

    def _handle_remote_gate_done(self, ev):
        msg_id = ev.payload["msg_id"]
        meta = self._inflight.pop(msg_id, None)
        if meta is None:
            return
        now = ev.time
        # record completion
        meta["end_time"] = now
        self._completed_metas.append(meta)
        self.stats["completed_transfers"] += 1
        self.stats["successful_two_qubit_gates"] += 1
        latency = now - meta["start_time"]
        self.stats["gate_latencies"].append(latency)
        if self.verbose:
            print(f"[t={now}] remote gate done for msg#{msg_id}; latency={latency}")

    # ----------------------
    # Simulation driver
    # ----------------------
    def run_single_trial(self):
        # Reset simulator state
        self._evq.clear()
        self._seq = 0
        self._inflight.clear()
        self._next_msg_id = 0
        self.edge_queues.clear()
        self.edge_busy_until.clear()
        self.edge_busy_count.clear()
        self._completed_metas.clear()

        # Reset stats
        for k in list(self.stats.keys()):
            if isinstance(self.stats[k], list):
                self.stats[k].clear()
            elif isinstance(self.stats[k], dict) or isinstance(self.stats[k], defaultdict):
                self.stats[k].clear()
            else:
                self.stats[k] = 0

        print("\n=== Starting Discrete-Event Simulation (slice waits for comm completion) ===")

        num_slices = len(self.timeslices)
        self.sim_time = 0.0

        for slice_idx, timeslice in enumerate(self.timeslices):
            # CRITICAL FIX: Wait for ALL previous events first (including migrations from previous slice)
            while self._evq or self._inflight:
                ev = self._pop_event()
                self._dispatch_event(ev)
            
            # NOW the slice can truly start - all previous work is done
            slice_start = self.sim_time
            
            if self.verbose:
                print(f"\n=== Timeslice {slice_idx} start @ DES time {slice_start}, {len(timeslice)} gates ===")

            mapping = self.partition[slice_idx]

            # Schedule all gates in this timeslice
            for gate in timeslice:
                qids = [q for q in gate.qubits]
                
                if len(qids) == 1:
                    # Local single-qubit gate
                    self._record_gate_latency(slice_start, self.gate_latency)
                    continue

                if len(qids) == 2:
                    q1, q2 = qids
                    node1 = mapping.get(q1)
                    node2 = mapping.get(q2)
                    
                    if node1 is None or node2 is None:
                        continue

                    if node1 == node2:
                        # Local two-qubit gate
                        self._record_gate_latency(slice_start, self.gate_latency)
                    else:
                        # Inter-core two-qubit gate
                        self.stats["total_two_qubit_gates"] += 1
                        self.stats["total_intercore_transfers"] += 1
                        
                        # Get path and log it
                        path_nodes = self._shortest_path_nodes(node1, node2)
                        if path_nodes:
                            path_info = self._get_path_info(path_nodes)
                            if self.verbose:
                                print(f"[t={slice_start}] Gate transfer {q1}->{q2}: {node1}->{node2}")
                                print(f"  Path: {' -> '.join(str(n) for n in path_nodes)}")
                                print(f"  Edge weights: {path_info['edge_weights']}")
                                print(f"  Total weighted distance: {path_info['total_weight']:.4f}")
                                print(f"  Number of hops: {path_info['num_hops']}")
                        
                        meta = self._create_transfer(gate, slice_start, node1, node2, slice_idx)
                        if meta:
                            meta["is_gate_transfer"] = True

            # Process all gate events until this slice's gates are done
            while self._evq or self._inflight:
                ev = self._pop_event()
                self._dispatch_event(ev)

            # Update sim_time to reflect when this slice completed
            if self._completed_metas:
                latest_end = max(meta.get("end_time", 0) for meta in self._completed_metas if meta.get("end_time"))
                if latest_end > self.sim_time:
                    self.sim_time = latest_end
            
            # If no transfers, advance by gate latency
            if self.sim_time == slice_start:
                self.sim_time = slice_start + self.gate_latency

            slice_end = self.sim_time
            
            if self.verbose:
                print(f"Timeslice {slice_idx} finished @ DES time {slice_end}")

            # Schedule inter-slice migrations AFTER this slice completes
            if slice_idx < num_slices - 1:
                curr_map = self.partition[slice_idx]
                next_map = self.partition[slice_idx + 1]
                
                for q in curr_map:
                    src_node = curr_map[q]
                    dst_node = next_map.get(q)
                    
                    if dst_node is not None and src_node != dst_node:
                        # Get path and log it
                        path_nodes = self._shortest_path_nodes(src_node, dst_node)
                        if path_nodes:
                            path_info = self._get_path_info(path_nodes)
                            if self.verbose:
                                print(f"[t={self.sim_time}] Migration of qubit {q}: {src_node}->{dst_node}")
                                print(f"  Path: {' -> '.join(str(n) for n in path_nodes)}")
                                print(f"  Edge weights: {path_info['edge_weights']}")
                                print(f"  Total weighted distance: {path_info['total_weight']:.4f}")
                                print(f"  Number of hops: {path_info['num_hops']}")
                        
                        transfer_meta = self._create_transfer(
                            gate=f"move_{q}",
                            slice_start=self.sim_time,
                            node_src=src_node,
                            node_dst=dst_node,
                            slice_idx=slice_idx,
                        )
                        
                        if transfer_meta:
                            transfer_meta["is_timeslice_transfer"] = True
            
            # DON'T process migrations here - they'll be processed at the start of next iteration

        # Final cleanup
        while self._evq or self._inflight:
            ev = self._pop_event()
            self._dispatch_event(ev)

        self.stats["total_sim_time"] = self.sim_time

        if self.verbose:
            print(f"\n=== Simulation completed at t={self.sim_time} ===")
            print(f"Total two-qubit gates: {self.stats['total_two_qubit_gates']}")
            print(f"Successful transfers: {self.stats['completed_transfers']}")
            print(f"Failed transfers: {self.stats['failed_two_qubit_gates']}")
            print(f"Total transfer attempts: {self.stats['transfer_attempts']}")
            print(f"Successful hops: {self.stats['successful_hops']}")
            print(f"Failed hops: {self.stats['failed_hops']}")

        return self.stats


    def _get_path_info(self, path_nodes):
        """Get detailed path information including weights."""
        if not path_nodes or len(path_nodes) < 2:
            return {
                'path': path_nodes,
                'num_hops': 0,
                'edge_weights': [],
                'total_weight': 0.0,
                'edge_success_probs': []
            }
        
        edge_weights = []
        edge_success_probs = []
        total_weight = 0.0
        
        for i in range(len(path_nodes) - 1):
            u = path_nodes[i]
            v = path_nodes[i + 1]
            
            weight = self.net[u][v].get('weight', 1.0)
            edge_weights.append(f"{u}->{v}: {weight:.4f}")
            total_weight += weight
            
            success_prob = self._edge_prob(u, v)
            edge_success_probs.append(f"{u}->{v}: {success_prob:.4f}")
        
        return {
            'path': path_nodes,
            'num_hops': len(path_nodes) - 1,
            'edge_weights': edge_weights,
            'total_weight': total_weight,
            'edge_success_probs': edge_success_probs
        }




    def _record_gate_latency(self, slice_start, latency):
        """Record latency for a completed local gate."""
        self.stats["gate_latencies"].append(latency)
        # Optionally update total_sim_time if you want to track simulation time
        if latency + slice_start > self.stats.get("total_sim_time", 0.0):
            self.stats["total_sim_time"] = latency + slice_start

    def _dispatch_event(self, ev: Event):
        # small dispatcher to call handlers
        if ev.etype == "execute_local_gate":
            self._handle_execute_local_gate(ev)
        elif ev.etype == "local_gate_done":
            self._handle_local_gate_done(ev)
        elif ev.etype == "start_hop":
            self._handle_start_hop(ev)
        elif ev.etype == "finish_hop":
            self._handle_finish_hop(ev)
        elif ev.etype == "transfer_arrived":
            self._handle_transfer_arrived(ev)
        elif ev.etype == "remote_gate_done":
            self._handle_remote_gate_done(ev)
        else:
            if self.verbose:
                print(f"[WARN] Unknown event {ev.etype} at t={ev.time}")

# End of file
