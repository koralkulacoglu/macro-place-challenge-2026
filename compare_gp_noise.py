
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
from submissions.koral.gp import run_global_placement as run_gp_orig
from submissions.koral.gp_alt import run_global_placement as run_gp_alt

def _load_benchmark(name: str):
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists():
        benchmark, plc = load_benchmark_from_dir(str(root))
        return benchmark, plc
    return None, None

def compare_gp_stochastic(bench_name="ibm01", n_steps=200):
    print(f"Loading {bench_name}...")
    benchmark, plc = _load_benchmark(bench_name)
    if not benchmark:
        print(f"Failed to load {bench_name}")
        return

    # Common parameters
    gp_params = {
        "pop_size": 4, # Use multiple replicas for PT ladder
        "n_steps": n_steps,
        "verbose": True,
        "log_every": 20,
        "seed": 42,
        "time_budget_s": 600.0
    }

    baseline_cache = Path("gp_baseline.pt")
    if baseline_cache.exists():
        print("\n>>> Loading Original GP Baseline from cache...")
        checkpoint = torch.load(baseline_cache)
        pos_orig = checkpoint['pos']
        hist_orig = checkpoint['hist']
        time_orig = checkpoint['time']
    else:
        print("\n>>> Running Original GP (Parallel Tempering)...")
        t0 = time.time()
        pos_orig, hist_orig = run_gp_orig(benchmark, plc, **gp_params)
        time_orig = time.time() - t0
        print(f">>> Caching baseline to {baseline_cache}...")
        torch.save({'pos': pos_orig, 'hist': hist_orig, 'time': time_orig}, baseline_cache)
    
    print("\n>>> Running Alt GP (PT + Langevin Noise)...")
    t0 = time.time()
    pos_alt, hist_alt = run_gp_alt(benchmark, plc, **gp_params)
    time_alt = time.time() - t0

    # Final evaluations via official oracle
    c_orig = compute_proxy_cost(torch.from_numpy(pos_orig).float(), benchmark, plc)
    c_alt = compute_proxy_cost(torch.from_numpy(pos_alt).float(), benchmark, plc)

    print("\n" + "="*55)
    print(f"STOCHASTIC COMPARISON ON {bench_name} ({n_steps} steps)")
    print("="*55)
    print(f"{'Metric':<20} | {'Original GP':<15} | {'Alt GP (Noise)':<15}")
    print("-" * 55)
    print(f"{'Time (s)':<20} | {time_orig:<15.2f} | {time_alt:<15.2f}")
    print(f"{'Proxy Cost':<20} | {c_orig['proxy_cost']:<15.4f} | {c_alt['proxy_cost']:<15.4f}")
    print(f"{'Wirelength':<20} | {c_orig['wirelength_cost']:<15.4f} | {c_alt['wirelength_cost']:<15.4f}")
    print(f"{'Density':<20} | {c_orig['density_cost']:<15.4f} | {c_alt['density_cost']:<15.4f}")
    print(f"{'Congestion':<20} | {c_orig['congestion_cost']:<15.4f} | {c_alt['congestion_cost']:<15.4f}")
    print(f"{'Overlaps':<20} | {c_orig['overlap_count']:<15} | {c_alt['overlap_count']:<15}")
    print("="*55)

    # Plotting
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Stochastic GP Comparison: {bench_name}", fontsize=16)

    # 1. Real Proxy Cost
    axes[0, 0].plot(hist_orig['real_proxy'], label='Original', alpha=0.7)
    axes[0, 0].plot(hist_alt['real_proxy'], label='Alt (Noise)', alpha=0.7)
    axes[0, 0].set_title("Real Proxy Cost (FastEvaluator)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # 2. Real Congestion
    axes[0, 1].plot(hist_orig['real_cong'], label='Original', alpha=0.7)
    axes[0, 1].plot(hist_alt['real_cong'], label='Alt (Noise)', alpha=0.7)
    axes[0, 1].set_title("Real Congestion")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # 3. Real Wirelength
    axes[1, 0].plot(hist_orig['real_wl'], label='Original', alpha=0.7)
    axes[1, 0].plot(hist_alt['real_wl'], label='Alt (Noise)', alpha=0.7)
    axes[1, 0].set_title("Real Wirelength")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # 4. Real Density
    axes[1, 1].plot(hist_orig['real_dens'], label='Original', alpha=0.7)
    axes[1, 1].plot(hist_alt['real_dens'], label='Alt (Noise)', alpha=0.7)
    axes[1, 1].set_title("Real Density")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = "gp_noise_comparison.png"
    plt.savefig(plot_path)
    print(f"\nComparison plot saved to {plot_path}")

if __name__ == "__main__":
    compare_gp_stochastic(n_steps=200)
