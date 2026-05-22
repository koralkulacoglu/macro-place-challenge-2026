
import os
import sys
import torch
import numpy as np
import time
from pathlib import Path

# Add current dir to sys.path so 'engine' is importable
sys.path.append(str(Path(__file__).parent))

from macro_place.loader import load_benchmark_from_dir
from submissions.koral.engine import StaticDesignData, FastEvaluator
from submissions.koral.placer import parallel_lahc_polish

def test_smoke_parallel():
    bench_name = "ibm01"
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / bench_name
    print(f"Loading {bench_name}...")
    benchmark, plc = load_benchmark_from_dir(str(root))
    
    print("Bootstrapping design data...")
    data = StaticDesignData.extract(benchmark, plc)
    
    init_pos = benchmark.macro_positions.numpy().astype(np.float64)
    
    print("Launching parallel LAHC smoke test (4 chains, 10 seconds)...")
    t0 = time.time()
    try:
        out = parallel_lahc_polish(
            benchmark,
            data,
            init_pos,
            list_len=10,
            time_budget_s=10.0,
            n_chains=4,
            base_seed=42,
            verbose=True
        )
        duration = time.time() - t0
        print(f"\nSUCCESS!")
        print(f"  Best Proxy: {out['proxy_cost']:.4f}")
        print(f"  Total Iters: {out['iters']}")
        print(f"  Wall time: {duration:.2f}s")
    except Exception as e:
        print(f"\nFAILED!")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    test_smoke_parallel()
