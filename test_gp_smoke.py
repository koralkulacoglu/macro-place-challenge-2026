
import torch
import numpy as np
from submissions.koral.gp import run_global_placement
from macro_place.benchmark import Benchmark

def test_gp_smoke():
    # Mock benchmark
    class MockBenchmark:
        def __init__(self):
            self.num_hard_macros = 2
            self.num_macros = 4
            self.num_nets = 2
            self.canvas_width = 100.0
            self.canvas_height = 100.0
            self.grid_cols = 10
            self.grid_rows = 10
            self.hroutes_per_micron = 1.0
            self.vroutes_per_micron = 1.0
            self.macro_sizes = torch.tensor([[10, 10], [10, 10], [5, 5], [5, 5]], dtype=torch.float32)
            self.macro_positions = torch.tensor([[20, 20], [80, 80], [50, 50], [60, 60]], dtype=torch.float32)
            self.macro_fixed = torch.tensor([False, False, False, False], dtype=torch.bool)
            self.port_positions = torch.zeros((0, 2), dtype=torch.float32)
            # Nets: net0 (macros 0, 1, 2), net1 (macros 2, 3)
            self.net_pin_nodes = [
                torch.tensor([[0, 0], [1, 0], [2, 0]], dtype=torch.long),
                torch.tensor([[2, 1], [3, 0]], dtype=torch.long)
            ]
            self.macro_pin_offsets = [torch.zeros((1, 2)) for _ in range(4)]
            self.net_nodes = [torch.tensor([0, 1, 2]), torch.tensor([2, 3])]

        def get_movable_mask(self):
            return ~self.macro_fixed

    bench = MockBenchmark()
    print("Running GP smoke test...")
    try:
        best_pos = run_global_placement(bench, pop_size=2, n_steps=10, verbose=True)
        print("GP smoke test PASSED")
        print("Best positions sample:", best_pos[0])
    except Exception as e:
        print("GP smoke test FAILED")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_gp_smoke()
