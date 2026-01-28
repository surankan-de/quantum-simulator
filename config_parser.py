from typing import Dict, List, Tuple
import os

def _parse_value(value: str):
    v = value.strip()
    try:
        return int(v)
    except Exception:
        pass
    if "," in v:
        parts = [p.strip() for p in v.split(',') if p.strip()]
        try:
            nums = [float(p) for p in parts]
            return nums if len(nums) > 1 else nums[0]
        except Exception:
            return parts
    return v


def parse_config_file(path: str) -> Dict:
    assert os.path.exists(path), f"Config file not found: {path}"
    config = {}
    with open(path, 'r') as f:
        for ln in f:
            ln = ln.strip()
            if not ln or ln.startswith('#'):
                continue
            if '=' not in ln:
                continue
            k, v = ln.split('=', 1)
            key = k.strip()
            val = _parse_value(v)
            config[key] = val

    assert 'mode' in config, "Missing required parameter: mode"
    assert config['mode'] in ['simulate'], "mode must be 'simulate' or ..."

    assert 'topology' in config, "Missing required parameter: topology"
    assert config['topology'] in ['mesh', 'line', 'star', 'tree', 'torus', 'complete'], "Invalid topology, must be one of mesh, line, star, tree, torus, complete"

    assert 'num_cores_x' in config, "Missing required parameter: num_cores_x"
    assert 'num_cores_y' in config, "Missing required parameter: num_cores_y"
    assert 'ncm_qubits' in config, "Missing required parameter: ncm_qubits"
    assert 'ncp_qubits' in config, "Missing required parameter: ncp_qubits"

    assert 'edge_weight_range' in config, "Missing required parameter: edge_weight_range"
    assert 'circuit_folders' in config, "Missing required parameter: circuit_folders"
    assert 'placers' in config, "Missing required parameter: placers"
    assert 'repeats' in config, "Missing required parameter: repeats"
    
    assert 'out_csv' in config, "Missing required parameter: out_csv"
    assert 'plot_file' in config, "Missing required parameter: plot_file"
    ewr = config.get('edge_weight_range')
    if isinstance(ewr, list) and len(ewr) >= 2:
        config['edge_weight_range'] = (float(ewr[0]), float(ewr[1]))
    else:
        raise AssertionError("edge_weight_range must be two numbers separated by a comma, e.g. 1.0,10.0")

    cf = config.get('circuit_folders')
    if isinstance(cf, list):
        folders = cf
    elif isinstance(cf, str):
        folders = [p.strip() for p in cf.split(',') if p.strip()]
    else:
        folders = list(cf)

    validated_folders: List[str] = []
    for p in folders:
        pp = os.path.expanduser(p)
        if not os.path.isdir(pp):
            print(f"Warning: circuit folder does not exist: {pp}")
        validated_folders.append(pp)
    config['circuit_folders'] = validated_folders

    raw_placers = config.get('placers')
    if isinstance(raw_placers, str):
        pls = [p.strip() for p in raw_placers.split(',') if p.strip()]
    elif isinstance(raw_placers, list):
        pls = [str(p).strip() for p in raw_placers]
    else:
        pls = [str(raw_placers).strip()]
    config['placers'] = pls
    
    return config