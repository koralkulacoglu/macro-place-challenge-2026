
import torch
import numpy as np
import os
import sys
import time

# Add project root to sys.path
sys.path.insert(0, os.getcwd())

from macro_place.loader import load_benchmark_from_dir
from submissions.koral.placer import GraphGradPlacer
import submissions.koral.placer as placer_mod

def test_initial_movement(bench_names=["ibm01", "ibm02", "ibm03"]):
    results = []
    
    # Intercept _build_population_seeds to see the very first placements.
    original_build_seeds = placer_mod._build_population_seeds
    captured_seeds = []
    
    def mocked_build_seeds(*args, **kwargs):
        seeds = original_build_seeds(*args, **kwargs)
        captured_seeds.append(seeds.copy())
        return seeds
    
    placer_mod._build_population_seeds = mocked_build_seeds
    
    try:
        for bench_name in bench_names:
            bench_dir = f"external/MacroPlacement/Testcases/ICCAD04/{bench_name}"
            if not os.path.exists(bench_dir):
                print(f"Benchmark directory {bench_dir} not found. Skipping.")
                continue

            print(f"\n--- Testing {bench_name} ---")
            benchmark, plc = load_benchmark_from_dir(bench_dir)
            n_hard = benchmark.num_hard_macros
            initial_hard_pos = benchmark.macro_positions[:n_hard].clone()
            
            # Test lock_hard=True (Default)
            placer_locked = GraphGradPlacer(lock_hard=True, n_restarts=1, soft_steps=1, verbose=False)
            res_locked = placer_locked.place(benchmark)
            moved_locked = torch.sqrt(torch.sum((res_locked[:n_hard] - initial_hard_pos)**2, dim=1))
            avg_locked = torch.mean(moved_locked).item()
            
            # Test lock_hard=False (Joint)
            captured_seeds.clear()
            placer_joint = GraphGradPlacer(lock_hard=False, pop_size=16, n_epochs=0, steps_per_epoch=0, verbose=False)
            try:
                # We just want to trigger seed generation
                placer_joint.place(benchmark)
            except Exception:
                # Might fail due to 0 epochs but seeds should be captured
                pass
            
            avg_seed_dist = 0.0
            if captured_seeds:
                seeds = captured_seeds[0]
                seed_hard_pos = torch.tensor(seeds[:, :n_hard, :])
                init_expanded = initial_hard_pos.unsqueeze(0).expand(seeds.shape[0], -1, -1)
                seed_movements = torch.sqrt(torch.sum((seed_hard_pos - init_expanded)**2, dim=2))
                avg_seed_dist = torch.mean(seed_movements).item()
            
            results.append({
                "bench": bench_name,
                "n_hard": n_hard,
                "avg_locked_move": avg_locked,
                "avg_seed_move": avg_seed_dist
            })
            
            print(f"  Avg Move (Legalization): {avg_locked:.6f} um")
            print(f"  Avg Move (Seeds):        {avg_seed_dist:.6f} um")

    finally:
        placer_mod._build_population_seeds = original_build_seeds

    print("\n" + "="*60)
    print(f"{'Benchmark':<10} | {'Hard Macros':<10} | {'Avg Move (Leg)':<15} | {'Avg Move (Seeds)':<15}")
    print("-"*60)
    for r in results:
        print(f"{r['bench']:<10} | {r['n_hard']:<10} | {r['avg_locked_move']:<15.6f} | {r['avg_seed_move']:<15.6f}")
    print("="*60)

if __name__ == "__main__":
    test_initial_movement()
