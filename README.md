# Quantum Multicore Placer Simulator

A small Python toolkit to evaluate placer algorithms that map quantum circuits (QASM) onto a multicore architecture. The simulator compares average runtime and expected inter-core communication for multiple placer implementations.

**Language:** Python

**Entry point:** main.py

- Run a simulation using the example config:

```powershell
python main.py --config example_config.txt
```


**Configuration**

Configuration is read from a simple key=value text file by `config_parser.py`. See `example_config.txt` for a working sample. Required keys:

- `mode`: must be `simulate`
- `topology`: one of `mesh`, `line`, `star`, `tree`, `torus`, `complete`
- `num_cores_x`: integer, cores in X dimension
- `num_cores_y`: integer, cores in Y dimension
- `ncm_qubits`: integer, number of communication qubits per core
- `ncp_qubits`: integer, number of per-core qubits
- `edge_weight_range`: two numbers separated by comma (e.g. `1.0,3.0`)
- `circuit_folders`: comma-separated folders containing `.qasm` files
- `placers`: comma-separated list of placers to run (see below)
- `repeats`: integer, number of runs per placer (averaging)
- `out_csv`: output path for CSV results
- `plot_file`: output path for the comparison PNG

`config_parser.py` validates and normalizes these values before `main.py` runs.

**Placer Algorithms**

Available placers (files in the repo): `TRHQA`, `GTRHQA`, `HQA`, `QUBO`, `RANDOM`, `BFS`.

- TRHQA_placer.py
- GTRHQA_placer.py
- HQA_placer.py
- Qubo_placer.py
- Random_placer.py
- BFS_placer.py

Each placer implements placement and cost interfaces used by `main.py` (e.g., `place_per_timeslice`, `per_timeslice_cost`).

**Circuits**

Place `.qasm` circuit files under the `circuits/` tree. Point `circuit_folders` in the config to the folders you want processed.

**Outputs**

- A CSV with averaged runtime and cost values (path = `out_csv`).
- A comparison PNG plot (path = `plot_file`).


