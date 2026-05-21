
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent))

from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from submissions.koral.placer import FastEvaluator
from submissions.koral.gp import rudy_demand, _smooth_1d_along_axis, _build_pin_tensors

def _load_plc(name: str):
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    ng45 = {
        "ariane133": "ariane133", "ariane136": "ariane136",
        "nvdla": "nvdla", "mempool_tile": "mempool_tile",
    }
    d = ng45.get(name.replace("_ng45", ""))
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists():
            _, plc = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))
            return plc
    return None

def compare_congestion(bench_name):
    print(f"\n--- Comparing Congestion for {bench_name} ---")
    plc = _load_plc(bench_name)
    if not plc:
        print(f"Failed to load {bench_name}")
        return

    from macro_place.benchmark import Benchmark
    # We need a Benchmark object for FastEvaluator
    # The load_benchmark functions return (benchmark, plc)
    # Re-loading to get the benchmark object properly
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / bench_name
    if root.exists():
        benchmark, _ = load_benchmark_from_dir(str(root))
    else:
        d = bench_name.replace("_ng45", "")
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        benchmark, _ = load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))

    # 1. Official Oracle Congestion
    cost_official = compute_proxy_cost(benchmark.macro_positions, benchmark, plc)
    cong_official = cost_official["congestion_cost"]
    print(f"Official Oracle Congestion: {cong_official:.6f}")

    # 2. FastEvaluator "Bit-Perfect" Port
    fe = FastEvaluator(benchmark, plc)
    cong_fast = fe._congestion_cost()
    print(f"FastEvaluator Port Congestion: {cong_fast:.6f}")
    diff_fast = abs(cong_official - cong_fast)
    print(f"Difference (Official vs Fast): {diff_fast:.2e}")

    # 3. RUDY Approximation (from gp.py)
    device = torch.device("cpu")
    owner_idx, pin_off, net_id, n_nets = _build_pin_tensors(benchmark, device)
    pop = benchmark.macro_positions.unsqueeze(0).to(device)
    port_pos = benchmark.port_positions.to(device)
    
    v_dem, h_dem = rudy_demand(
        pop, owner_idx, pin_off, net_id, n_nets, port_pos,
        benchmark.num_macros, int(benchmark.grid_cols), int(benchmark.grid_rows),
        float(benchmark.canvas_width), float(benchmark.canvas_height),
        float(benchmark.hroutes_per_micron), float(benchmark.vroutes_per_micron)
    )
    
    # Apply smoothing like in the proxy
    v_s = _smooth_1d_along_axis(v_dem, 2, 0)
    h_s = _smooth_1d_along_axis(h_dem, 2, 1)
    
    # Add macro congestion (demand from macros themselves)
    # In gp.py this is often omitted or handled differently, but let's see
    # The official proxy adds macro routing demand.
    # FastEvaluator handles it via self.v_macro_cong and self.h_macro_cong
    
    vm = torch.from_numpy(fe.v_macro_cong).unsqueeze(0) / fe.grid_v_routes
    hm = torch.from_numpy(fe.h_macro_cong).unsqueeze(0) / fe.grid_h_routes
    
    combined = torch.cat([(v_s + vm).reshape(1, -1), (h_s + hm).reshape(1, -1)], dim=1)
    sorted_c, _ = torch.sort(combined, descending=True)
    cnt = int(sorted_c.shape[1] * 0.05)
    cong_rudy = sorted_c[0, :cnt].mean().item() if cnt > 0 else sorted_c[0, 0].item()
    
    print(f"RUDY (from gp.py) Congestion: {cong_rudy:.6f}")
    diff_rudy = abs(cong_official - cong_rudy)
    print(f"Difference (Official vs RUDY): {diff_rudy:.2e}")

if __name__ == "__main__":
    benchmarks = ["ibm01", "ariane133"]
    for b in benchmarks:
        try:
            compare_congestion(b)
        except Exception as e:
            print(f"Error comparing {b}: {e}")
            import traceback
            traceback.print_exc()
