
import os
import sys
import time
import numpy as np
import torch
from pathlib import Path
from scipy.stats import pearsonr

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent))

from macro_place.loader import load_benchmark, load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from submissions.koral.placer import FastEvaluator
from submissions.koral.gp import rudy_demand, _smooth_1d_along_axis, _build_pin_tensors

from tqdm import tqdm

def _load_plc(name: str):
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        _, plc = load_benchmark_from_dir(str(root))
        return plc
    return None

def run_correlation_test(bench_name="ibm01", n_trials=100):
    print(f"Loading {bench_name}...")
    plc = _load_plc(bench_name)
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / bench_name
    benchmark, _ = load_benchmark_from_dir(str(root))
    
    fe = FastEvaluator(benchmark, plc)
    
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    n_macros = benchmark.num_macros
    sizes = benchmark.macro_sizes.numpy()
    
    device = torch.device("cpu")
    owner_idx, pin_off, net_id, n_nets = _build_pin_tensors(benchmark, device)
    port_pos = benchmark.port_positions.to(device)
    
    official_scores = []
    fast_scores = []
    rudy_scores = []
    dfg_scores = []
    
    print(f"Starting {n_trials} random trials...")
    
    for i in tqdm(range(n_trials), desc="Correlating"):
        # Generate random placement (within bounds)
        rand_pos = np.zeros((n_macros, 2), dtype=np.float64)
        for m in range(n_macros):
            hw, hh = sizes[m] / 2
            rand_pos[m, 0] = np.random.uniform(hw, cw - hw)
            rand_pos[m, 1] = np.random.uniform(hh, ch - hh)
        
        # 1. Update FastEvaluator and get its score
        # We can just overwrite the whole state for a clean comparison
        fe.positions = rand_pos.copy()
        fe._init_caches() # Re-init everything for the new positions
        cong_fast = fe._congestion_cost()
        
        # 2. Official Oracle
        rand_pos_t = torch.from_numpy(rand_pos).float()
        cost_official = compute_proxy_cost(rand_pos_t, benchmark, plc)
        cong_official = cost_official["congestion_cost"]
        
        # 3. RUDY Approximation
        pop = rand_pos_t.unsqueeze(0).to(device)
        v_dem, h_dem = rudy_demand(
            pop, owner_idx, pin_off, net_id, n_nets, port_pos,
            n_macros, int(benchmark.grid_cols), int(benchmark.grid_rows),
            cw, ch,
            float(benchmark.hroutes_per_micron), float(benchmark.vroutes_per_micron)
        )
        v_s = _smooth_1d_along_axis(v_dem, 2, 0)
        h_s = _smooth_1d_along_axis(h_dem, 2, 1)
        vm = torch.from_numpy(fe.v_macro_cong).unsqueeze(0) / fe.grid_v_routes
        hm = torch.from_numpy(fe.h_macro_cong).unsqueeze(0) / fe.grid_h_routes
        combined = torch.cat([(v_s + vm).reshape(1, -1), (h_s + hm).reshape(1, -1)], dim=1)
        sorted_c, _ = torch.sort(combined, descending=True)
        cnt = int(sorted_c.shape[1] * 0.05)
        cong_rudy = sorted_c[0, :cnt].mean().item() if cnt > 0 else sorted_c[0, 0].item()
        
        # 4. Dual-Field Gaussian (DFG) Model
        # This is a smoothed version of the FastEvaluator's discrete logic
        # We'll use the fe grids and apply a Gaussian blur to simulate the DFG GP field
        from scipy.ndimage import gaussian_filter
        h_pin = fe.h_pin_cong / fe.grid_h_routes
        v_pin = fe.v_pin_cong / fe.grid_v_routes
        h_mac = fe.h_macro_cong / fe.grid_h_routes
        v_mac = fe.v_macro_cong / fe.grid_v_routes
        
        # Gaussian blur (sigma=1.0 grid cells)
        h_dfg = gaussian_filter(h_pin + h_mac, sigma=1.0)
        v_dfg = gaussian_filter(v_pin + v_mac, sigma=1.0)
        
        combined_dfg = np.concatenate([h_dfg.ravel(), v_dfg.ravel()])
        cnt_dfg = int(combined_dfg.size * 0.05)
        cong_dfg = np.sort(combined_dfg)[::-1][:cnt_dfg].mean()
        
        official_scores.append(cong_official)
        fast_scores.append(cong_fast)
        rudy_scores.append(cong_rudy)
        dfg_scores.append(cong_dfg)

    official_scores = np.array(official_scores)
    fast_scores = np.array(fast_scores)
    rudy_scores = np.array(rudy_scores)
    dfg_scores = np.array(dfg_scores)
    
    corr_fast, _ = pearsonr(official_scores, fast_scores)
    corr_rudy, _ = pearsonr(official_scores, rudy_scores)
    corr_dfg, _ = pearsonr(official_scores, dfg_scores)
    
    mae_fast = np.mean(np.abs(official_scores - fast_scores))
    mae_rudy = np.mean(np.abs(official_scores - rudy_scores))
    mae_dfg = np.mean(np.abs(official_scores - dfg_scores))
    
    print("\n" + "="*40)
    print(f"RESULTS FOR {bench_name} ({n_trials} trials)")
    print("="*40)
    print(f"FastEvaluator (Bit-Perfect) vs Official:")
    print(f"  Pearson Correlation: {corr_fast:.8f}")
    print(f"  Mean Absolute Error: {mae_fast:.8e}")
    print(f"\nRUDY (gp.py) vs Official:")
    print(f"  Pearson Correlation: {corr_rudy:.8f}")
    print(f"  Mean Absolute Error: {mae_rudy:.8e}")
    print(f"\nDual-Field Gaussian (DFG) vs Official:")
    print(f"  Pearson Correlation: {corr_dfg:.8f}")
    print(f"  Mean Absolute Error: {mae_dfg:.8e}")
    print("="*40)


if __name__ == "__main__":
    run_correlation_test()
