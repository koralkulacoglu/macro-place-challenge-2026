
import os
import sys
import torch
import numpy as np
from pathlib import Path

# Add the project root to sys.path
sys.path.append(str(Path(__file__).parent))

from macro_place.loader import load_benchmark_from_dir

def check_seed_diversity(bench_name="ibm01", n_populations=10, K=4):
    print(f"Loading {bench_name}...")
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / bench_name
    benchmark, _ = load_benchmark_from_dir(str(root))
    
    cw = float(benchmark.canvas_width)
    ch = float(benchmark.canvas_height)
    diag = math.sqrt(cw**2 + ch**2)
    n_macros = benchmark.num_macros
    
    # We'll use the same logic as in run_global_placement to generate seeds
    init_pos = benchmark.macro_positions.float() # [N, 2]
    
    populations = []
    print(f"Generating {n_populations} populations of size K={K}...")
    
    for seed in range(n_populations):
        torch.manual_seed(seed)
        rng = torch.Generator().manual_seed(seed)
        
        # Jitter logic from gp.py
        pop = init_pos.unsqueeze(0).expand(K, -1, -1).clone()
        jitter_levels = torch.linspace(0.0, 0.10, K)
        
        for k in range(K):
            if jitter_levels[k] > 0:
                scale = float(jitter_levels[k]) * min(cw, ch)
                pop[k] = pop[k] + torch.randn(n_macros, 2, generator=rng) * scale
        
        populations.append(pop.numpy())

    # Calculate differences
    # 1. Diversity WITHIN a population (Replicas 0 vs K-1)
    within_diffs = []
    for p in populations:
        dist = np.linalg.norm(p[0] - p[K-1], axis=1).mean()
        within_diffs.append(dist)
    
    avg_within = np.mean(within_diffs)
    
    # 2. Diversity ACROSS different seeds (Replica 0 vs Replica 0)
    across_diffs = []
    for i in range(n_populations):
        for j in range(i + 1, n_populations):
            dist = np.linalg.norm(populations[i][0] - populations[j][0], axis=1).mean()
            across_diffs.append(dist)
    
    avg_across = np.mean(across_diffs) if across_diffs else 0

    # 3. Maximum spread across everything
    all_stacked = np.concatenate(populations, axis=0) # [P*K, N, 2]
    total_dist = []
    for i in range(len(all_stacked)):
        for j in range(i + 1, len(all_stacked)):
            dist = np.linalg.norm(all_stacked[i] - all_stacked[j], axis=1).mean()
            total_dist.append(dist)
    
    avg_total = np.mean(total_dist)

    print("\n" + "="*50)
    print(f"SEED DIVERSITY ANALYSIS: {bench_name}")
    print("="*50)
    print(f"Canvas Diagonal: {diag:.2f}")
    print(f"Max Jitter Scale: {0.10 * min(cw, ch):.2f} (10% of min dimension)")
    print("-" * 50)
    print(f"Avg Distance (Cold vs Hottest Replica): {avg_within:.2f} ({avg_within/diag*100:.1f}% of diagonal)")
    print(f"Avg Distance (Between different seeds): {avg_across:.2f} ({avg_across/diag*100:.1f}% of diagonal)")
    print(f"Avg Global Pairwise Diversity:          {avg_total:.2f} ({avg_total/diag*100:.1f}% of diagonal)")
    print("="*50)

if __name__ == "__main__":
    import math
    check_seed_diversity()
