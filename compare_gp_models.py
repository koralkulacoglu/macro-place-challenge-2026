
import os
import sys
import time
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent))

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost
from submissions.koral.gp import run_global_placement

def _load_benchmark(name: str):
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        benchmark, plc = load_benchmark_from_dir(str(root))
        return benchmark, plc
    return None, None

def compare_gp_models(bench_name="ibm01", n_steps=200):
    print(f"Loading {bench_name}...")
    benchmark, plc = _load_benchmark(bench_name)
    if not benchmark:
        print(f"Failed to load {bench_name}")
        return

    # Common parameters
    gp_params = {
        "pop_size": 1, # Use single replica for cleaner history comparison
        "n_steps": n_steps,
        "verbose": True,
        "log_every": 20,
        "seed": 42,
        "time_budget_s": 600.0 # Increase budget to see full 500 steps
    }

    print("\n>>> Running RUDY (Baseline) GP...")
    t0 = time.time()
    pos_rudy, hist_rudy = run_global_placement(benchmark, plc, use_dfg=False, **gp_params)
    time_rudy = time.time() - t0
    
    print("\n>>> Running DFG (Proposed) GP...")
    t0 = time.time()
    pos_dfg, hist_dfg = run_global_placement(benchmark, plc, use_dfg=True, **gp_params)
    time_dfg = time.time() - t0

    # Final evaluations via official oracle
    c_rudy = compute_proxy_cost(torch.from_numpy(pos_rudy).float(), benchmark, plc)
    c_dfg = compute_proxy_cost(torch.from_numpy(pos_dfg).float(), benchmark, plc)

    print("\n" + "="*50)
    print(f"FINAL COMPARISON ON {bench_name} ({n_steps} steps)")
    print("="*50)
    print(f"{'Metric':<20} | {'RUDY (Old)':<12} | {'DFG (New)':<12}")
    print("-" * 50)
    print(f"{'Time (s)':<20} | {time_rudy:<12.2f} | {time_dfg:<12.2f}")
    print(f"{'Proxy Cost':<20} | {c_rudy['proxy_cost']:<12.4f} | {c_dfg['proxy_cost']:<12.4f}")
    print(f"{'Wirelength':<20} | {c_rudy['wirelength_cost']:<12.4f} | {c_dfg['wirelength_cost']:<12.4f}")
    print(f"{'Density':<20} | {c_rudy['density_cost']:<12.4f} | {c_dfg['density_cost']:<12.4f}")
    print(f"{'Congestion':<20} | {c_rudy['congestion_cost']:<12.4f} | {c_dfg['congestion_cost']:<12.4f}")
    print(f"{'Overlaps':<20} | {c_rudy['overlap_count']:<12} | {c_dfg['overlap_count']:<12}")
    print("="*50)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"GP Real Cost Evolution (FastEvaluator): {bench_name}", fontsize=16)

    # 1. Real Proxy Cost
    axes[0, 0].plot(hist_rudy['real_proxy'], label='RUDY', alpha=0.8)
    axes[0, 0].plot(hist_dfg['real_proxy'], label='DFG', alpha=0.8)
    axes[0, 0].set_title("Official Proxy Cost")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Real Congestion
    axes[0, 1].plot(hist_rudy['real_cong'], label='RUDY', alpha=0.8)
    axes[0, 1].plot(hist_dfg['real_cong'], label='DFG', alpha=0.8)
    axes[0, 1].set_title("Official Congestion")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Real Wirelength
    axes[1, 0].plot(hist_rudy['real_wl'], label='RUDY', alpha=0.8)
    axes[1, 0].plot(hist_dfg['real_wl'], label='DFG', alpha=0.8)
    axes[1, 0].set_title("Official Wirelength")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Real Density
    axes[1, 1].plot(hist_rudy['real_dens'], label='RUDY', alpha=0.8)
    axes[1, 1].plot(hist_dfg['real_dens'], label='DFG', alpha=0.8)
    axes[1, 1].set_title("Official Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = "gp_comparison.png"
    plt.savefig(plot_path)
    print(f"\nPlot saved to {plot_path}")

if __name__ == "__main__":
    compare_gp_models(n_steps=500)
