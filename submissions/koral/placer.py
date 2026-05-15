"""
KoralPlacer — DREAMPlace + SA polish for the Partcl Macro Placement Challenge.

Pipeline:
  1. Convert Benchmark → Bookshelf files (temp dir)
  2. Run DREAMPlace (GPU, macro_place_flag=1, legalize_flag=1)
  3. Recover positions from PlaceDB as center-coord tensor
  4. SA polish: use compute_proxy_cost as oracle for ~N seconds

Usage:
  uv run evaluate submissions/koral/placer.py -b ibm01
  uv run evaluate submissions/koral/placer.py --all
"""

import sys
import os
import math
import random
import time
import tempfile
import json
import logging
import numpy as np
import torch
from pathlib import Path

# ── DREAMPlace paths (inside Docker container) ────────────────────────────────
_DP_INSTALL = "/opt/dreamplace"
_DP_PKG     = "/opt/dreamplace/dreamplace"
for _p in [_DP_INSTALL, _DP_PKG]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Suppress DREAMPlace's verbose logging
logging.getLogger().setLevel(logging.WARNING)

from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost

# Import bookshelf writer from same directory
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from bookshelf import write_bookshelf, dreamplace_nodes_to_tensor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_plc(benchmark: Benchmark):
    """Reload PlacementCost for a benchmark — needed to call compute_proxy_cost."""
    try:
        from macro_place.loader import load_benchmark_from_dir, load_benchmark

        # IBM ICCAD04 path
        root = Path("external/MacroPlacement/Testcases/ICCAD04") / benchmark.name
        if root.exists():
            _, plc = load_benchmark_from_dir(str(root))
            return plc

        # NG45 paths
        ng45_map = {
            "ariane133": "ariane133", "ariane136": "ariane136",
            "nvdla": "nvdla",         "mempool_tile": "mempool_tile",
        }
        ng45_name = ng45_map.get(benchmark.name)
        if ng45_name:
            base = (Path("external/MacroPlacement/Flows/NanGate45")
                    / ng45_name / "netlist" / "output_CT_Grouping")
            if (base / "netlist.pb.txt").exists():
                _, plc = load_benchmark(
                    str(base / "netlist.pb.txt"),
                    str(base / "initial.plc"),
                )
                return plc
    except Exception as e:
        print(f"[warn] Could not load PlacementCost for {benchmark.name}: {e}")
    return None


def _dreamplace_available() -> bool:
    try:
        import dreamplace.ops.place_io.place_io  # noqa: F401
        return True
    except ImportError:
        return False


# ── Main placer ───────────────────────────────────────────────────────────────

class KoralPlacer:
    def __init__(
        self,
        target_density: float = 0.0,     # 0 = auto from utilization
        density_weight: float = 8e-5,
        gamma: float = 4.0,
        sa_time_budget: int = 240,        # seconds for SA polish (0 = skip)
        seed: int = 42,
    ):
        self.target_density  = target_density
        self.density_weight  = density_weight
        self.gamma           = gamma
        self.sa_time_budget  = sa_time_budget
        self.seed            = seed

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)
        np.random.seed(self.seed)

        plc = _load_plc(benchmark)

        if _dreamplace_available():
            placement = self._run_dreamplace(benchmark)
        else:
            print("[warn] DREAMPlace not available — falling back to initial positions")
            placement = benchmark.macro_positions.clone()

        if plc is not None and self.sa_time_budget > 0:
            placement = self._sa_polish(placement, benchmark, plc)

        return placement

    # ── DREAMPlace stage ──────────────────────────────────────────────────────

    def _run_dreamplace(self, benchmark: Benchmark) -> torch.Tensor:
        import Params
        import PlaceDB
        import NonLinearPlace

        # Compute target density from benchmark area utilization
        target_density = self.target_density
        if target_density == 0.0:
            macro_area = (benchmark.macro_sizes[:, 0] * benchmark.macro_sizes[:, 1]).sum().item()
            canvas_area = benchmark.canvas_width * benchmark.canvas_height
            utilization = macro_area / canvas_area
            # Add headroom — too tight causes legalization failure
            target_density = min(0.95, utilization + 0.2)

        with tempfile.TemporaryDirectory(prefix=f"koral_{benchmark.name}_") as tmpdir:
            # Write Bookshelf files
            movable_hard = [i for i in range(benchmark.num_hard_macros)
                            if not benchmark.macro_fixed[i]]
            fixed_hard   = [i for i in range(benchmark.num_hard_macros)
                            if benchmark.macro_fixed[i]]
            movable_soft = list(range(benchmark.num_hard_macros, benchmark.num_macros))
            ordered = movable_hard + movable_soft + fixed_hard

            aux_path = write_bookshelf(benchmark, tmpdir)

            # Build DREAMPlace params
            params_dict = {
                "aux_input":          aux_path,
                "gpu":                1 if torch.cuda.is_available() else 0,
                "target_density":     target_density,
                "density_weight":     self.density_weight,
                "gamma":              self.gamma,
                "macro_place_flag":   1,
                "legalize_flag":        1,
                "abacus_legalize_flag": 1,  # site_h=1nm so all heights are valid multiples
                "detailed_place_flag":  0,
                "global_place_flag":  1,
                "enable_fillers":     1,
                "stop_overflow":      0.07,
                "gp_noise_ratio":     0.025,
                "random_center_init_flag": 1,
                "ignore_net_degree":  100,
                "num_threads":        8,
                "random_seed":        self.seed,
                "scale_factor":       0.0,
                "result_dir":         os.path.join(tmpdir, "results"),
                "global_place_stages": [
                    {
                        "num_bins_x": 128, "num_bins_y": 128,
                        "iteration": 500, "learning_rate": 0.01,
                        "wirelength": "weighted_average",
                        "optimizer": "nesterov",
                        "Llambda_density_weight_iteration": 1,
                        "Lsub_iteration": 1,
                    },
                    {
                        "num_bins_x": 512, "num_bins_y": 512,
                        "iteration": 1000, "learning_rate": 0.01,
                        "wirelength": "weighted_average",
                        "optimizer": "nesterov",
                        "Llambda_density_weight_iteration": 1,
                        "Lsub_iteration": 1,
                    },
                ],
                "macro_halo_x": 50,   # 50nm gap → prevents float-precision boundary overlaps
                "macro_halo_y": 50,
                "plot_flag":    0,
                "dtype":        "float32",
            }

            params_path = os.path.join(tmpdir, "params.json")
            with open(params_path, "w") as f:
                json.dump(params_dict, f)

            # Load params and run
            params = Params.Params()
            params.load(params_path)

            placedb = PlaceDB.PlaceDB()
            placedb(params)

            placer = NonLinearPlace.NonLinearPlace(params, placedb, timer=None)
            placer(params, placedb, learning_rate_value=None)

            # Recover positions
            num_ports = benchmark.port_positions.shape[0]
            placement = dreamplace_nodes_to_tensor(
                placedb, ordered, fixed_hard, num_ports, benchmark
            )

        # Post-DREAMPlace: legalize hard macros to guarantee zero overlaps
        # (DREAMPlace's internal check may pass with tiny near-zero overlaps)
        placement = self._legalize_hard(placement, benchmark)

        return placement

    # ── Post-DREAMPlace legalization ──────────────────────────────────────────

    def _legalize_hard(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """
        Greedy min-displacement legalization of hard macros.
        Ensures zero overlaps with a 0.05μm safety gap.
        Adapted from will_seed._legalize.
        """
        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n].numpy()
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        pos = placement[:n].numpy().copy().astype(np.float64)

        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        order = sorted(range(n), key=lambda i: -sizes[i, 0] * sizes[i, 1])
        placed = np.zeros(n, dtype=bool)
        legal = pos.copy()

        for idx in order:
            if not movable[idx]:
                placed[idx] = True
                continue
            if placed.any():
                dx = np.abs(legal[idx, 0] - legal[:, 0])
                dy = np.abs(legal[idx, 1] - legal[:, 1])
                c = (dx < sep_x[idx] + 0.05) & (dy < sep_y[idx] + 0.05) & placed
                c[idx] = False
                if not c.any():
                    placed[idx] = True
                    continue
            step = max(sizes[idx, 0], sizes[idx, 1]) * 0.25
            best_p = legal[idx].copy()
            best_d = float('inf')
            for r in range(1, 200):
                found = False
                for dxm in range(-r, r + 1):
                    for dym in range(-r, r + 1):
                        if abs(dxm) != r and abs(dym) != r:
                            continue
                        cx = np.clip(pos[idx, 0] + dxm * step, half_w[idx], cw - half_w[idx])
                        cy = np.clip(pos[idx, 1] + dym * step, half_h[idx], ch - half_h[idx])
                        if placed.any():
                            dx2 = np.abs(cx - legal[:, 0])
                            dy2 = np.abs(cy - legal[:, 1])
                            c = (dx2 < sep_x[idx] + 0.05) & (dy2 < sep_y[idx] + 0.05) & placed
                            c[idx] = False
                            if c.any():
                                continue
                        d = (cx - pos[idx, 0]) ** 2 + (cy - pos[idx, 1]) ** 2
                        if d < best_d:
                            best_d = d
                            best_p = np.array([cx, cy])
                            found = True
                if found:
                    break
            legal[idx] = best_p
            placed[idx] = True

        result = placement.clone()
        result[:n] = torch.tensor(legal, dtype=torch.float32)
        return result

    # ── SA polish stage ───────────────────────────────────────────────────────

    def _sa_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
    ) -> torch.Tensor:
        """
        Short SA refinement using the actual TILOS proxy cost as the objective.
        Only runs for sa_time_budget seconds; accepts the best result seen.
        """
        n_hard   = benchmark.num_hard_macros
        cw, ch   = benchmark.canvas_width, benchmark.canvas_height
        sizes    = benchmark.macro_sizes.numpy()
        half_w   = sizes[:, 0] / 2
        half_h   = sizes[:, 1] / 2
        movable  = benchmark.get_movable_mask().numpy()
        movable_idx = [i for i in range(benchmark.num_macros) if movable[i]]

        if not movable_idx:
            return placement

        # Pre-compute minimum separation for overlap checking
        sep_x = (sizes[:n_hard, 0:1] + sizes[:n_hard, 0:1].T) / 2
        sep_y = (sizes[:n_hard, 1:2] + sizes[:n_hard, 1:2].T) / 2

        def macro_overlaps(idx, pos_np):
            """O(N) check: does macro idx overlap any other hard macro?"""
            dx = np.abs(pos_np[idx, 0] - pos_np[:n_hard, 0])
            dy = np.abs(pos_np[idx, 1] - pos_np[:n_hard, 1])
            mask = (dx < sep_x[idx] + 0.01) & (dy < sep_y[idx] + 0.01)
            mask[idx] = False
            return mask.any()

        pos = placement.numpy().copy()
        current_cost = compute_proxy_cost(
            torch.tensor(pos, dtype=torch.float32), benchmark, plc
        )["proxy_cost"]
        best_pos  = pos.copy()
        best_cost = current_cost

        T      = max(cw, ch) * 0.05
        T_end  = max(cw, ch) * 0.001
        deadline = time.time() + self.sa_time_budget
        step = 0

        print(f"  [SA polish] starting cost={current_cost:.4f}, budget={self.sa_time_budget}s")

        while time.time() < deadline:
            i   = random.choice(movable_idx)
            old = pos[i].copy()
            shift = T * 0.3
            pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift), half_w[i], cw - half_w[i])
            pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift), half_h[i], ch - half_h[i])

            # Fast O(N) overlap pre-check before the expensive cost eval
            if i < n_hard and macro_overlaps(i, pos):
                pos[i] = old
                continue

            new_cost = compute_proxy_cost(
                torch.tensor(pos, dtype=torch.float32), benchmark, plc
            )["proxy_cost"]
            delta = new_cost - current_cost

            if delta < 0 or random.random() < math.exp(-delta / max(T, 1e-10)):
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_pos  = pos.copy()
            else:
                pos[i] = old

            step += 1
            frac = min(1.0, (deadline - time.time()) / self.sa_time_budget)
            T = T_end + (T - T_end) * frac

        print(f"  [SA polish] {step} steps, final best={best_cost:.4f}")
        return torch.tensor(best_pos, dtype=torch.float32)
