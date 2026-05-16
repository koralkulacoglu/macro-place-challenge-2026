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
# Unbuffered output for live progress in nohup/pipe contexts
sys.stdout.reconfigure(line_buffering=True)

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
        sa_time_budget: int = 600,        # seconds for CD+LNS+oracle SA polish
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

        # Start from CT positions, then try DREAMPlace if available.
        # CT positions: already near-optimal from Circuit Training (good congestion/density).
        ct = benchmark.macro_positions.clone()
        ct_legal = self._legalize_hard(ct, benchmark)

        placement = ct_legal  # default: CT positions

        if _dreamplace_available():
            # Try DREAMPlace variants and keep the best result.
            # center_init=True: good for small/sparse benchmarks (good global spread from scratch)
            # center_init=False: good for large/dense benchmarks (refine CT positions)
            ct_cost = compute_proxy_cost(ct_legal, benchmark, plc)["proxy_cost"] if plc else float('inf')
            best_dp_cost = ct_cost
            best_placement = ct_legal

            # Run center-init DREAMPlace. For benchmarks where center beats CT, try a
            # second seed to reduce non-determinism (DREAMPlace GPU results vary slightly).
            seeds_to_try = [self.seed]
            for seed_i, dp_seed in enumerate(seeds_to_try):
                tag = "center" if seed_i == 0 else f"center-s{seed_i+1}"
                try:
                    dp = self._run_dreamplace(benchmark, center_init=True,
                                              fix_soft=False, dp_seed=dp_seed)
                    dp = self._legalize_hard(dp, benchmark)
                    if self._count_hard_overlaps_f32(dp, benchmark) > 0:
                        continue
                    if plc is None:
                        continue
                    dp_cost = compute_proxy_cost(dp, benchmark, plc)["proxy_cost"]
                    if dp_cost < best_dp_cost:
                        print(f"  [start] DREAMPlace({tag}) {dp_cost:.4f} < best {best_dp_cost:.4f}")
                        best_dp_cost = dp_cost
                        best_placement = dp
                        # Try 2nd seed only when center beats CT (worth 3 extra minutes)
                        if dp_cost < ct_cost and self.seed + 1 not in seeds_to_try:
                            seeds_to_try.append(self.seed + 1)
                    else:
                        print(f"  [start] DREAMPlace({tag}) {dp_cost:.4f} >= best {best_dp_cost:.4f}")
                except Exception as e:
                    print(f"  [start] DREAMPlace({tag}) failed: {e}")

            placement = best_placement

        if plc is not None and self.sa_time_budget > 0:
            placement = self._cd_lns_polish(placement, benchmark, plc)

        return placement

    # ── DREAMPlace stage ──────────────────────────────────────────────────────

    @staticmethod
    def _patch_macro_legalize(n_hard_movable: int):
        """
        Monkey-patch DREAMPlace's macro legalization to only process hard macros.
        The Hannan-grid search for soft macros is extremely slow on large benchmarks.
        Since our Bookshelf ordering puts hard macros first (indices 0..n_hard_movable-1),
        temporarily reducing num_movable_nodes to n_hard_movable limits legalization to
        hard macros only, letting soft macros stay at their global-placement positions.
        """
        try:
            import dreamplace.ops.macro_legalize.macro_legalize as _ml
            if getattr(_ml.MacroLegalize, '_koral_patched', False):
                _ml.MacroLegalize._koral_n_hard = n_hard_movable
                return
            _orig_call = _ml.MacroLegalize.__call__

            def _fast_call(self, init_pos, pos):
                n_hard = getattr(_ml.MacroLegalize, '_koral_n_hard', None)
                if n_hard is not None and n_hard < self.num_movable_nodes:
                    orig_n = self.num_movable_nodes
                    self.num_movable_nodes = n_hard
                    try:
                        return _orig_call(self, init_pos, pos)
                    finally:
                        self.num_movable_nodes = orig_n
                return _orig_call(self, init_pos, pos)

            _ml.MacroLegalize.__call__ = _fast_call
            _ml.MacroLegalize._koral_patched = True
            _ml.MacroLegalize._koral_n_hard = n_hard_movable
            print(f"  [patch] macro_legalize limited to {n_hard_movable} hard macros")
        except Exception as e:
            print(f"  [patch] macro_legalize patch failed: {e} — legalization may be slow")

    def _run_dreamplace(self, benchmark: Benchmark, center_init: bool = True,
                        fix_soft: bool = False, dp_seed: int = None) -> torch.Tensor:
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
            movable_hard = [i for i in range(benchmark.num_hard_macros)
                            if not benchmark.macro_fixed[i]]
            fixed_hard   = [i for i in range(benchmark.num_hard_macros)
                            if benchmark.macro_fixed[i]]
            movable_soft = list(range(benchmark.num_hard_macros, benchmark.num_macros))
            # DREAMPlace ordering: movable_hard first, then movable_soft, then fixed_hard.
            # The monkey-patch below limits macro legalization to the first
            # len(movable_hard) nodes (hard macros only), skipping the slow Hannan-grid
            # search for thousands of soft macros.
            ordered = movable_hard + movable_soft + fixed_hard

            aux_path = write_bookshelf(benchmark, tmpdir, fix_soft=fix_soft)

            # Build DREAMPlace params
            params_dict = {
                "aux_input":          aux_path,
                "gpu":                1 if torch.cuda.is_available() else 0,
                "target_density":     target_density,
                "density_weight":     self.density_weight,
                "gamma":              self.gamma,
                "macro_place_flag":   0,  # skip Hannan grid; global placement spreads macros first
                "legalize_flag":        0,
                "abacus_legalize_flag": 0,
                "detailed_place_flag":  0,
                "global_place_flag":  1,
                "enable_fillers":     1,
                "stop_overflow":      0.07,
                "gp_noise_ratio":     0.025,
                "random_center_init_flag": 1 if center_init else 0,
                "ignore_net_degree":  100,
                "num_threads":        8,
                "random_seed":        dp_seed if dp_seed is not None else self.seed,
                "scale_factor":       0.0,
                "result_dir":         os.path.join(tmpdir, "results"),
                "global_place_stages": [
                    {
                        "num_bins_x": 64, "num_bins_y": 64,
                        "iteration": 1000, "learning_rate": 0.01,
                        "wirelength": "weighted_average",
                        "optimizer": "nesterov",
                        "Llambda_density_weight_iteration": 1,
                        "Lsub_iteration": 1,
                    },
                    {
                        "num_bins_x": 256, "num_bins_y": 256,
                        "iteration": 1500, "learning_rate": 0.01,
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

        # Inner legalization: resolve DREAMPlace micro-overlaps so the outer legalization
        # starts from a better state (important for dense benchmarks where DREAMPlace
        # leaves many small overlaps that a single 2000-pass outer legalization may not resolve).
        placement = self._legalize_hard(placement, benchmark)
        return placement

    # ── Post-DREAMPlace legalization ──────────────────────────────────────────

    @staticmethod
    def _count_hard_overlaps_f32(placement: torch.Tensor, benchmark: Benchmark) -> int:
        """Count overlapping hard macro pairs using float32 (matches evaluator precision)."""
        n = benchmark.num_hard_macros
        pos = placement[:n].numpy().astype(np.float32)
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float32)
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        dx = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
        dy = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
        ov = (sep_x - dx > 0) & (sep_y - dy > 0)
        np.fill_diagonal(ov, False)
        return int(ov.sum()) // 2

    def _legalize_hard(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """
        Fast O(N² × passes) push-based legalization of hard macros.
        Each pass scans all pairs; overlapping pairs are resolved by pushing
        the movable macro in the minimum-displacement direction (x or y).
        Converges in a few passes for typical DREAMPlace outputs.
        """
        GAP = 0.005  # 5nm gap to ensure float32 conversion doesn't re-introduce overlaps

        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n].numpy().astype(bool)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        pos = placement[:n].numpy().copy().astype(np.float64)

        # Clamp all movable macros into bounds first
        for i in range(n):
            if movable[i]:
                pos[i, 0] = np.clip(pos[i, 0], half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1], half_h[i], ch - half_h[i])

        # Exact separations for overlap detection; gap-inflated for resolution
        sep_x_exact = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y_exact = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        sep_x = sep_x_exact + GAP
        sep_y = sep_y_exact + GAP

        pass_idx = 0
        for pass_idx in range(2000):
            # Vectorized overlap detection (any positive physical overlap)
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break

            # Sequential (Gauss-Seidel) push — handles boundary clamping correctly:
            # when one macro is clamped, the full push goes to its unclamped neighbor.
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not movable[i] and not movable[j]:
                        continue
                    dx = abs(pos[i, 0] - pos[j, 0])
                    dy = abs(pos[i, 1] - pos[j, 1])
                    ox = sep_x[i, j] - dx
                    oy = sep_y[i, j] - dy
                    if ox <= 0 or oy <= 0:
                        continue
                    both = movable[i] and movable[j]
                    if ox <= oy:
                        need = ox
                        sign = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                        if both:
                            xi_new = np.clip(pos[i, 0] + sign * need / 2, half_w[i], cw - half_w[i])
                            xj_new = np.clip(pos[j, 0] - sign * need / 2, half_w[j], cw - half_w[j])
                            got_i  = abs(xi_new - pos[i, 0])
                            got_j  = abs(xj_new - pos[j, 0])
                            if got_i + got_j < need - 1e-9:
                                xi_new = np.clip(pos[i, 0] + sign * (need - got_j), half_w[i], cw - half_w[i])
                                xj_new = np.clip(pos[j, 0] - sign * (need - got_i), half_w[j], cw - half_w[j])
                        else:
                            xi_new = np.clip(pos[i, 0] + sign * need, half_w[i], cw - half_w[i]) if movable[i] else pos[i, 0]
                            xj_new = np.clip(pos[j, 0] - sign * need, half_w[j], cw - half_w[j]) if movable[j] else pos[j, 0]
                        if abs(xi_new - pos[i, 0]) > 1e-9 or abs(xj_new - pos[j, 0]) > 1e-9:
                            changed = True
                        pos[i, 0] = xi_new; pos[j, 0] = xj_new
                    else:
                        need = oy
                        sign = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                        if both:
                            yi_new = np.clip(pos[i, 1] + sign * need / 2, half_h[i], ch - half_h[i])
                            yj_new = np.clip(pos[j, 1] - sign * need / 2, half_h[j], ch - half_h[j])
                            got_i  = abs(yi_new - pos[i, 1])
                            got_j  = abs(yj_new - pos[j, 1])
                            if got_i + got_j < need - 1e-9:
                                yi_new = np.clip(pos[i, 1] + sign * (need - got_j), half_h[i], ch - half_h[i])
                                yj_new = np.clip(pos[j, 1] - sign * (need - got_i), half_h[j], ch - half_h[j])
                        else:
                            yi_new = np.clip(pos[i, 1] + sign * need, half_h[i], ch - half_h[i]) if movable[i] else pos[i, 1]
                            yj_new = np.clip(pos[j, 1] - sign * need, half_h[j], ch - half_h[j]) if movable[j] else pos[j, 1]
                        if abs(yi_new - pos[i, 1]) > 1e-9 or abs(yj_new - pos[j, 1]) > 1e-9:
                            changed = True
                        pos[i, 1] = yi_new; pos[j, 1] = yj_new
            if not changed:
                break

        # Diagonal push stage for any remaining stuck pairs:
        # push in BOTH x and y to break cycles that the min-direction approach can't escape
        for diag_pass in range(200):
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break
            changed = False
            for i in range(n):
                for j in range(i + 1, n):
                    if not is_ov[i, j] or (not movable[i] and not movable[j]):
                        continue
                    dx = abs(pos[i, 0] - pos[j, 0])
                    dy = abs(pos[i, 1] - pos[j, 1])
                    ox = sep_x[i, j] - dx
                    oy = sep_y[i, j] - dy
                    if ox <= 0 and oy <= 0:
                        continue
                    both = movable[i] and movable[j]
                    sx = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                    sy = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                    fx = max(ox, 0) * sx; fy = max(oy, 0) * sy
                    s = 0.5 if both else 1.0
                    if movable[i]:
                        xi_new = np.clip(pos[i, 0] + fx * s, half_w[i], cw - half_w[i])
                        yi_new = np.clip(pos[i, 1] + fy * s, half_h[i], ch - half_h[i])
                        if abs(xi_new-pos[i,0]) > 1e-9 or abs(yi_new-pos[i,1]) > 1e-9:
                            changed = True
                        pos[i, 0] = xi_new; pos[i, 1] = yi_new
                    if movable[j]:
                        xj_new = np.clip(pos[j, 0] - fx * s, half_w[j], cw - half_w[j])
                        yj_new = np.clip(pos[j, 1] - fy * s, half_h[j], ch - half_h[j])
                        if abs(xj_new-pos[j,0]) > 1e-9 or abs(yj_new-pos[j,1]) > 1e-9:
                            changed = True
                        pos[j, 0] = xj_new; pos[j, 1] = yj_new
            if not changed:
                break

        # Report remaining overlaps for debugging
        dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
        dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
        ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
        np.fill_diagonal(ov, False)
        n_ov = int(ov.sum()) // 2
        if n_ov > 0:
            print(f"  [legalize] {pass_idx+1}+diag passes, {n_ov} hard macro overlaps remain")
        result = placement.clone()
        result[:n] = torch.tensor(pos, dtype=torch.float32)
        return result

    # ── HPWL data structures ─────────────────────────────────────────────────

    def _build_hpwl_data(self, benchmark: Benchmark, pos: np.ndarray):
        """
        Precompute connectivity data for fast incremental HPWL in SA.

        Returns:
            macro_nets: list[list[int]] — for each movable hard macro, net IDs it belongs to
            net_hard:   list[list[int]] — for each net, movable hard macro indices in it
            net_fbbox:  np.ndarray [num_nets, 4] — fixed-node bbox (min_x,max_x,min_y,max_y)
            net_wts:    np.ndarray [num_nets] — net weights
        """
        n_hard  = benchmark.num_hard_macros
        n_mac   = benchmark.num_macros
        prt_pos = benchmark.port_positions.numpy()  # [n_ports, 2]
        movable = benchmark.get_movable_mask().numpy()

        # Unified position array: macros (0..n_mac) then ports (n_mac..)
        all_pos = np.vstack([pos, prt_pos]) if prt_pos.shape[0] > 0 else pos

        num_nets = benchmark.num_nets
        macro_nets: list = [[] for _ in range(n_hard)]
        net_hard:   list = []
        # Fixed-node bbox for each net (from soft macros + ports + fixed hard macros)
        net_fbbox = np.full((num_nets, 4), fill_value=np.inf)
        net_fbbox[:, 1] = -np.inf
        net_fbbox[:, 3] = -np.inf

        for k in range(num_nets):
            nodes = benchmark.net_nodes[k].numpy()
            hard_in = []
            for n in nodes.tolist():
                if n < n_hard and movable[n]:
                    hard_in.append(n)
                    macro_nets[n].append(k)
                else:
                    # Fixed node — contribute to fixed bbox
                    px = float(all_pos[n, 0])
                    py = float(all_pos[n, 1])
                    if px < net_fbbox[k, 0]: net_fbbox[k, 0] = px
                    if px > net_fbbox[k, 1]: net_fbbox[k, 1] = px
                    if py < net_fbbox[k, 2]: net_fbbox[k, 2] = py
                    if py > net_fbbox[k, 3]: net_fbbox[k, 3] = py
            net_hard.append(hard_in)

        return macro_nets, net_hard, net_fbbox, benchmark.net_weights.numpy()

    @staticmethod
    def _hpwl_nets(net_ids, pos, net_hard, net_fbbox, net_wts):
        """Compute weighted HPWL for the given subset of nets."""
        total = 0.0
        for k in net_ids:
            fb   = net_fbbox[k]
            mnx  = fb[0]; mxx = fb[1]; mny = fb[2]; mxy = fb[3]
            for h in net_hard[k]:
                px = pos[h, 0]; py = pos[h, 1]
                if px < mnx: mnx = px
                if px > mxx: mxx = px
                if py < mny: mny = py
                if py > mxy: mxy = py
            if mnx <= mxx:
                total += net_wts[k] * ((mxx - mnx) + (mxy - mny))
        return total

    def _update_soft_macros(self, pos: np.ndarray, benchmark: Benchmark) -> np.ndarray:
        """
        Move each soft macro to the centroid of its connected hard macros and ports.
        One pass of force-directed placement for soft macros; fast (O(total_pins)).
        """
        n_hard = benchmark.num_hard_macros
        n_mac  = benchmark.num_macros
        port_pos = benchmark.port_positions.numpy()
        movable  = benchmark.get_movable_mask().numpy()
        cw, ch   = benchmark.canvas_width, benchmark.canvas_height
        half_w   = benchmark.macro_sizes[:, 0].numpy() / 2
        half_h   = benchmark.macro_sizes[:, 1].numpy() / 2

        # Unified position: macros[0..n_mac) then ports[n_mac..)
        all_pos = np.vstack([pos, port_pos]) if port_pos.shape[0] > 0 else pos

        # Accumulate weighted centroid for each soft macro from non-soft nodes
        n_soft  = n_mac - n_hard
        sum_x   = np.zeros(n_soft)
        sum_y   = np.zeros(n_soft)
        weight  = np.zeros(n_soft)

        net_wts = benchmark.net_weights.numpy()
        for k in range(benchmark.num_nets):
            nodes = benchmark.net_nodes[k].numpy().tolist()
            soft_in = [n for n in nodes if n_hard <= n < n_mac and movable[n]]
            other   = [n for n in nodes if n < n_hard or n >= n_mac]
            if not soft_in or not other:
                continue
            other_pos = all_pos[other]
            cx = float(other_pos[:, 0].mean())
            cy = float(other_pos[:, 1].mean())
            wt = float(net_wts[k])
            for m in soft_in:
                idx = m - n_hard
                sum_x[idx]  += wt * cx
                sum_y[idx]  += wt * cy
                weight[idx] += wt

        for s in range(n_soft):
            m = n_hard + s
            if not movable[m] or weight[s] == 0:
                continue
            pos[m, 0] = np.clip(sum_x[s] / weight[s], half_w[m], cw - half_w[m])
            pos[m, 1] = np.clip(sum_y[s] / weight[s], half_h[m], ch - half_h[m])

        return pos

    # ── CD + LNS polish stage ─────────────────────────────────────────────────

    def _cd_lns_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
    ) -> torch.Tensor:
        """
        Coordinate descent + LNS polish on hard macros.

        Inner loop (fast): for each movable macro, tries 20 Gaussian-shifted candidate
        positions + 3 orientation flips and accepts whichever minimises delta-HPWL.
        delta-HPWL is O(degree) using precomputed "fixed bbox" per (net, macro) pair.

        Outer loop: evaluates full proxy-cost oracle once per CD pass (~0.5s).
        Reverts positions to best-known if oracle doesn't confirm improvement.

        LNS: periodically tries all pairwise position-swaps within spatial clusters of
        k=8 macros; verifies with oracle.

        Final step: tries updating soft macros to centroids; reverts if no gain.
        """
        n_hard = benchmark.num_hard_macros
        n_mac  = benchmark.num_macros
        pp     = benchmark.port_positions.numpy()
        n_ports = pp.shape[0]
        cw, ch  = float(benchmark.canvas_width), float(benchmark.canvas_height)
        sizes   = benchmark.macro_sizes.numpy()
        hw, hh  = sizes[:, 0] / 2, sizes[:, 1] / 2
        movable = benchmark.get_movable_mask().numpy()
        movable_hard = [i for i in range(n_hard) if movable[i]]
        n_mv = len(movable_hard)
        nw   = benchmark.net_weights.numpy()
        poffs = benchmark.macro_pin_offsets  # list[Tensor[num_pins, 2]] per hard macro
        has_poffs = len(poffs) > 0

        if not movable_hard:
            return placement

        pos    = placement.numpy().copy()
        # Current orientation per hard macro: 0=N, 1=FN(flipX pins), 2=FS(flipY), 3=S(flipXY)
        ori    = np.zeros(n_hard, dtype=np.int8)

        # ── Precompute pin-level net data ─────────────────────────────────────
        # all_cx/cy: positions of ALL nodes (hard macros, soft macros, ports)
        all_cx = np.zeros(n_mac + n_ports)
        all_cy = np.zeros(n_mac + n_ports)
        all_cx[:n_mac] = pos[:n_mac, 0]
        all_cy[:n_mac] = pos[:n_mac, 1]
        if n_ports > 0:
            all_cx[n_mac:] = pp[:, 0]
            all_cy[n_mac:] = pp[:, 1]

        # Precompute pin offsets for hard macros (base offsets before orientation)
        # pin_ox[i] = np.array of x-offsets for macro i's pins (one per pin)
        pin_ox = []
        pin_oy = []
        if has_poffs:
            for i in range(n_hard):
                if i < len(poffs) and poffs[i].numel() > 0:
                    pin_ox.append(poffs[i][:, 0].numpy())
                    pin_oy.append(poffs[i][:, 1].numpy())
                else:
                    pin_ox.append(np.zeros(1))
                    pin_oy.append(np.zeros(1))
        else:
            for i in range(n_hard):
                pin_ox.append(np.zeros(1))
                pin_oy.append(np.zeros(1))

        def _oriented_offsets(i, o):
            """Return (ox_arr, oy_arr) for macro i with orientation o."""
            ox = pin_ox[i].copy(); oy = pin_oy[i].copy()
            if o == 1: ox = -ox
            elif o == 2: oy = -oy
            elif o == 3: ox = -ox; oy = -oy
            return ox, oy

        # Build net pin list using net_pin_nodes if available, else net_nodes
        use_pinlevel = len(benchmark.net_pin_nodes) > 0 and has_poffs

        # net_entries[k] = list of (unified_node_idx, base_ox, base_oy, hard_macro_idx_or_-1)
        # For hard macros: hard_macro_idx = owner_idx; for others: -1
        net_entries = []
        macro_to_nets = [[] for _ in range(n_hard)]  # k for each movable hard macro

        for k in range(benchmark.num_nets):
            if use_pinlevel and k < len(benchmark.net_pin_nodes):
                pn = benchmark.net_pin_nodes[k]
                entries = []
                for row in pn.tolist():
                    owner, slot = row
                    if owner < n_hard:
                        ox = float(pin_ox[owner][slot]) if slot < len(pin_ox[owner]) else 0.0
                        oy = float(pin_oy[owner][slot]) if slot < len(pin_oy[owner]) else 0.0
                        entries.append((owner, ox, oy, owner))
                    elif owner < n_mac:
                        entries.append((owner, 0.0, 0.0, -1))
                    else:
                        entries.append((n_mac + (owner - n_mac), 0.0, 0.0, -1))
            else:
                nodes = benchmark.net_nodes[k].tolist()
                entries = []
                for owner in nodes:
                    if owner < n_mac:
                        entries.append((owner, 0.0, 0.0, owner if owner < n_hard else -1))
                    else:
                        entries.append((n_mac + (owner - n_mac), 0.0, 0.0, -1))

            if len(entries) < 2:
                net_entries.append(None)
                continue
            net_entries.append(entries)

            seen = set()
            for (_, _, _, mi) in entries:
                if mi >= 0 and movable[mi] and mi not in seen:
                    macro_to_nets[mi].append(k)
                    seen.add(mi)

        # Precompute "fixed bbox" for each (net k, hard macro i):
        # bbox of all pins in net k EXCLUDING macro i's pins.
        # Stored as flat arrays for speed.
        # fix_bbox[k][i] = (min_x, max_x, min_y, max_y)
        # We store per-macro dicts keyed by k.
        macro_fix_bbox = [{} for _ in range(n_hard)]  # macro_fix_bbox[i][k] = (x0,x1,y0,y1)

        def _compute_fix_bbox(k, excl_mi):
            entries = net_entries[k]
            if entries is None:
                return None
            xs, ys = [], []
            for (node, ox, oy, mi) in entries:
                if mi == excl_mi:
                    continue  # skip this macro's pins
                # Pin pos: for hard macro apply orientation, else use position
                if mi >= 0 and mi != excl_mi:
                    aox, aoy = _oriented_offsets(mi, ori[mi])
                    # Find which slot this is? We stored base offsets so just use ox,oy
                    # Apply orientation to stored ox,oy
                    aox_v = ox; aoy_v = oy
                    o = ori[mi]
                    if o == 1: aox_v = -aox_v
                    elif o == 2: aoy_v = -aoy_v
                    elif o == 3: aox_v = -aox_v; aoy_v = -aoy_v
                    xs.append(all_cx[node] + aox_v)
                    ys.append(all_cy[node] + aoy_v)
                else:
                    xs.append(all_cx[node] + ox)
                    ys.append(all_cy[node] + oy)
            if not xs:
                return (0.0, 0.0, 0.0, 0.0)
            return (min(xs), max(xs), min(ys), max(ys))

        # Build fix_bbox for all (macro, net) pairs
        for i in movable_hard:
            for k in macro_to_nets[i]:
                fb = _compute_fix_bbox(k, i)
                if fb is not None:
                    macro_fix_bbox[i][k] = fb

        def _rebuild_fix_bbox_for_nets(nets_to_update):
            """Recompute fix_bbox for all hard macros in the given nets."""
            for k in nets_to_update:
                entries = net_entries[k]
                if entries is None:
                    continue
                for (_, _, _, mi) in entries:
                    if mi >= 0 and movable[mi]:
                        fb = _compute_fix_bbox(k, mi)
                        if fb is not None:
                            macro_fix_bbox[mi][k] = fb

        def _delta_hpwl(i, new_cx, new_cy, new_o):
            """O(degree) delta-HPWL for moving macro i to (new_cx, new_cy) with orient new_o."""
            delta = 0.0
            old_cx, old_cy = all_cx[i], all_cy[i]
            for k in macro_to_nets[i]:
                fb = macro_fix_bbox[i].get(k)
                if fb is None:
                    continue
                fx0, fx1, fy0, fy1 = fb

                # Compute old and new bounding box contribution from macro i's pins
                entries = net_entries[k]
                old_min_x = old_max_x = old_min_y = old_max_y = None
                new_min_x = new_max_x = new_min_y = new_max_y = None

                for (node, ox, oy, mi) in entries:
                    if mi != i:
                        continue
                    # Old position with old orientation
                    aox_o, aoy_o = ox, oy
                    o_old = int(ori[i])
                    if o_old == 1: aox_o = -aox_o
                    elif o_old == 2: aoy_o = -aoy_o
                    elif o_old == 3: aox_o = -aox_o; aoy_o = -aoy_o
                    px_o, py_o = old_cx + aox_o, old_cy + aoy_o

                    # New position with new orientation
                    aox_n, aoy_n = ox, oy
                    if new_o == 1: aox_n = -aox_n
                    elif new_o == 2: aoy_n = -aoy_n
                    elif new_o == 3: aox_n = -aox_n; aoy_n = -aoy_n
                    px_n, py_n = new_cx + aox_n, new_cy + aoy_n

                    if old_min_x is None or px_o < old_min_x: old_min_x = px_o
                    if old_max_x is None or px_o > old_max_x: old_max_x = px_o
                    if old_min_y is None or py_o < old_min_y: old_min_y = py_o
                    if old_max_y is None or py_o > old_max_y: old_max_y = py_o

                    if new_min_x is None or px_n < new_min_x: new_min_x = px_n
                    if new_max_x is None or px_n > new_max_x: new_max_x = px_n
                    if new_min_y is None or py_n < new_min_y: new_min_y = py_n
                    if new_max_y is None or py_n > new_max_y: new_max_y = py_n

                if old_min_x is None:
                    continue  # macro i not in this net (shouldn't happen)

                old_wl = (max(fx1, old_max_x) - min(fx0, old_min_x) +
                          max(fy1, old_max_y) - min(fy0, old_min_y)) * nw[k]
                new_wl = (max(fx1, new_max_x) - min(fx0, new_min_x) +
                          max(fy1, new_max_y) - min(fy0, new_min_y)) * nw[k]
                delta += new_wl - old_wl
            return delta

        def _accept_move(i, new_cx, new_cy, new_o):
            """Apply move and update all_cx/cy and fix_bboxes for affected nets."""
            all_cx[i] = new_cx; all_cy[i] = new_cy
            pos[i, 0] = new_cx; pos[i, 1] = new_cy
            ori[i] = new_o
            # Rebuild fix_bbox for all macros in all nets containing i
            affected_nets = macro_to_nets[i]
            _rebuild_fix_bbox_for_nets(affected_nets)

        # Overlap check matrices
        _sg = 0.005
        sep_x = (sizes[:n_hard, 0:1] + sizes[:n_hard, 0:1].T) / 2 + _sg
        sep_y = (sizes[:n_hard, 1:2] + sizes[:n_hard, 1:2].T) / 2 + _sg

        def _overlaps(i, cx, cy):
            dx = np.abs(cx - pos[:n_hard, 0]); dy = np.abs(cy - pos[:n_hard, 1])
            mask = (dx < sep_x[i]) & (dy < sep_y[i]); mask[i] = False
            return bool(mask.any())

        def _rebuild_state(ref_pos, ref_ori):
            pos[:] = ref_pos; ori[:] = ref_ori
            all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
            for i in movable_hard:
                for k in macro_to_nets[i]:
                    fb = _compute_fix_bbox(k, i)
                    if fb is not None:
                        macro_fix_bbox[i][k] = fb

        # ── Initial oracle evaluation ─────────────────────────────────────────
        init_result = compute_proxy_cost(
            torch.tensor(pos, dtype=torch.float32), benchmark, plc
        )
        init_cost = init_result["proxy_cost"]
        _wl_frac = init_result["wirelength_cost"] / max(init_cost, 1e-9)
        best_cost = init_cost
        best_pos  = pos.copy()
        best_ori  = ori.copy()
        print(f"  [CD] initial={init_cost:.4f} (wl_frac={_wl_frac:.2%})")

        # Build congestion-guided macro weights (static snapshot after initial eval).
        # plc is now synced; query per-cell routing congestion and map movable macros to cells.
        _cong_weights = None   # None = uniform random selection (high-cong bias)
        _inv_weights   = None  # None = uniform random selection (low-cong bias for swap targets)
        try:
            if hasattr(benchmark, 'hard_macro_indices') and len(benchmark.hard_macro_indices) > 0:
                _plc_ids = [int(benchmark.hard_macro_indices[i]) for i in movable_hard]
                _h_cong  = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                _v_cong  = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                _total   = _h_cong + _v_cong
                _cells   = [plc.get_grid_cell_of_node(pid) for pid in _plc_ids]
                _mc      = np.array([_total[c] if 0 <= c < len(_total) else 0.0 for c in _cells])
                _thr     = float(np.percentile(_mc, 50))
                _wts     = np.maximum(0.0, _mc - _thr).astype(np.float64)
                if _wts.sum() > 0:
                    _cong_weights = _wts / _wts.sum()
                    # Inverse weights: prefer low-congestion macros for swap targets
                    _inv_wts = 1.0 / (_cong_weights + 1e-9)
                    _inv_weights = (_inv_wts / _inv_wts.sum()).astype(np.float64)
        except Exception:
            pass

        deadline = time.time() + self.sa_time_budget
        oracle_calls = 1
        pass_num = 0
        no_improve_streak = 0

        # ── Coordinate Descent ────────────────────────────────────────────────
        while time.time() < deadline:
            frac = max(0.0, (deadline - time.time()) / self.sa_time_budget)
            # Annealed move scale: start large, end small
            scale = max(cw, ch) * (0.08 * frac + 0.005)
            n_cands = 20

            order = random.sample(movable_hard, n_mv)
            n_accepted = 0

            for i in order:
                if time.time() >= deadline:
                    break

                best_delta = 0.0
                best_cx = pos[i, 0]; best_cy = pos[i, 1]; best_o = int(ori[i])
                found = False

                # Position candidates (Gaussian shifts)
                for _ in range(n_cands):
                    cx = float(np.clip(pos[i, 0] + random.gauss(0, scale), hw[i], cw - hw[i]))
                    cy = float(np.clip(pos[i, 1] + random.gauss(0, scale), hh[i], ch - hh[i]))
                    if _overlaps(i, cx, cy):
                        continue
                    d = _delta_hpwl(i, cx, cy, int(ori[i]))
                    if d < best_delta:
                        best_delta = d; best_cx = cx; best_cy = cy; best_o = int(ori[i]); found = True

                # Orientation flips at current position
                if has_poffs:
                    cur_cx, cur_cy = float(pos[i, 0]), float(pos[i, 1])
                    for o in (1, 2, 3):
                        d = _delta_hpwl(i, cur_cx, cur_cy, o)
                        if d < best_delta:
                            best_delta = d; best_cx = cur_cx; best_cy = cur_cy; best_o = o; found = True

                if found:
                    _accept_move(i, best_cx, best_cy, best_o)
                    n_accepted += 1

            if n_accepted == 0:
                break

            # ── Oracle call once per pass ─────────────────────────────────────
            cost = compute_proxy_cost(
                torch.tensor(pos, dtype=torch.float32), benchmark, plc
            )["proxy_cost"]
            oracle_calls += 1

            if cost < best_cost:
                best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                print(f"  [CD] pass {pass_num}: {cost:.4f}  ({n_accepted} accepted)")
                no_improve_streak = 0
            else:
                _rebuild_state(best_pos, best_ori)
                no_improve_streak += 1
                if no_improve_streak >= 5:
                    break  # HPWL gradient not correlated with proxy; stop wasting oracle calls

            pass_num += 1

        # ── LNS: pairwise swaps in spatial clusters ───────────────────────────
        # _overlaps_excl: check if macro i at (cx,cy) overlaps anything except macro excl
        def _overlaps_excl(i, cx, cy, excl):
            dx = np.abs(cx - pos[:n_hard, 0]); dy = np.abs(cy - pos[:n_hard, 1])
            mask = (dx < sep_x[i]) & (dy < sep_y[i])
            mask[i] = False; mask[excl] = False
            return bool(mask.any())

        # _swap_delta_hpwl: exact delta-HPWL for swapping positions of i and j
        # (handles shared nets correctly by temporarily applying both changes)
        def _swap_delta_hpwl(i, j):
            old_ci_x, old_ci_y = float(pos[i, 0]), float(pos[i, 1])
            old_cj_x, old_cj_y = float(pos[j, 0]), float(pos[j, 1])
            affected = set(macro_to_nets[i]) | set(macro_to_nets[j])
            delta = 0.0
            for k in affected:
                entries = net_entries[k]
                if entries is None:
                    continue
                # Old HPWL for net k
                old_xmin = old_xmax = old_ymin = old_ymax = None
                new_xmin = new_xmax = new_ymin = new_ymax = None
                for (node, ox, oy, mi) in entries:
                    # Old pin position
                    if mi == i:
                        o = int(ori[i])
                        aox, aoy = ox, oy
                        if o==1: aox=-aox
                        elif o==2: aoy=-aoy
                        elif o==3: aox,aoy=-aox,-aoy
                        px_o, py_o = old_ci_x+aox, old_ci_y+aoy
                        px_n, py_n = old_cj_x+aox, old_cj_y+aoy  # swapped pos
                    elif mi == j:
                        o = int(ori[j])
                        aox, aoy = ox, oy
                        if o==1: aox=-aox
                        elif o==2: aoy=-aoy
                        elif o==3: aox,aoy=-aox,-aoy
                        px_o, py_o = old_cj_x+aox, old_cj_y+aoy
                        px_n, py_n = old_ci_x+aox, old_ci_y+aoy  # swapped pos
                    else:
                        aox_e, aoy_e = ox, oy
                        if mi >= 0:
                            o = int(ori[mi])
                            if o==1: aox_e=-aox_e
                            elif o==2: aoy_e=-aoy_e
                            elif o==3: aox_e,aoy_e=-aox_e,-aoy_e
                        px_o = px_n = all_cx[node]+aox_e; py_o = py_n = all_cy[node]+aoy_e
                    if old_xmin is None or px_o<old_xmin: old_xmin=px_o
                    if old_xmax is None or px_o>old_xmax: old_xmax=px_o
                    if old_ymin is None or py_o<old_ymin: old_ymin=py_o
                    if old_ymax is None or py_o>old_ymax: old_ymax=py_o
                    if new_xmin is None or px_n<new_xmin: new_xmin=px_n
                    if new_xmax is None or px_n>new_xmax: new_xmax=px_n
                    if new_ymin is None or py_n<new_ymin: new_ymin=py_n
                    if new_ymax is None or py_n>new_ymax: new_ymax=py_n
                if old_xmin is not None:
                    old_wl = (old_xmax-old_xmin+old_ymax-old_ymin)*nw[k]
                    new_wl = (new_xmax-new_xmin+new_ymax-new_ymin)*nw[k]
                    delta += new_wl - old_wl
            return delta

        lns_k = 8
        lns_improvements = 0
        lns_no_swap_streak = 0
        # Cap LNS time adaptively based on WL fraction of initial proxy cost.
        # Low-WL benchmarks (ibm06, wl_frac=3%): 15% cap (90s) → oracle SA gets 460s+
        # to exploit improvements that oracle SA CAN find for such benchmarks.
        # High-WL benchmarks (ibm03+, wl_frac=20%+): up to 50% (300s) since LNS is
        # the primary optimizer for them. WL-dominated benchmarks that exit LNS
        # naturally before the cap are unaffected.
        lns_frac = max(0.15, min(0.50, _wl_frac * 3.0))
        lns_deadline = min(deadline - 2.0, time.time() + self.sa_time_budget * lns_frac)
        while time.time() < lns_deadline:
            center = random.choice(movable_hard)
            cx0, cy0 = pos[center, 0], pos[center, 1]
            dists = sorted([(abs(pos[j, 0]-cx0)+abs(pos[j, 1]-cy0), j)
                            for j in movable_hard if j != center])
            cluster = [center] + [idx for _, idx in dists[:lns_k - 1]]

            best_swap_delta = 0.0
            best_swap = None
            for ai in range(len(cluster)):
                for bi in range(ai + 1, len(cluster)):
                    i, j = cluster[ai], cluster[bi]
                    old_ci = (float(pos[i, 0]), float(pos[i, 1]))
                    old_cj = (float(pos[j, 0]), float(pos[j, 1]))
                    if _overlaps_excl(i, old_cj[0], old_cj[1], j) or \
                       _overlaps_excl(j, old_ci[0], old_ci[1], i):
                        continue
                    delta = _swap_delta_hpwl(i, j)
                    if delta < best_swap_delta:
                        best_swap_delta = delta; best_swap = (i, j, old_ci, old_cj)

            if best_swap is not None:
                i, j, old_ci, old_cj = best_swap
                _accept_move(i, old_cj[0], old_cj[1], int(ori[i]))
                _accept_move(j, old_ci[0], old_ci[1], int(ori[j]))
                lns_improvements += 1
                lns_no_swap_streak = 0
                # Verify with oracle every 10 accepted swaps
                if lns_improvements % 10 == 0 or time.time() > deadline - 10:
                    cost = compute_proxy_cost(
                        torch.tensor(pos, dtype=torch.float32), benchmark, plc
                    )["proxy_cost"]
                    oracle_calls += 1
                    if cost < best_cost:
                        best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                        print(f"  [LNS] {cost:.4f}")
                    else:
                        _rebuild_state(best_pos, best_ori)
            else:
                lns_no_swap_streak += 1
                if lns_no_swap_streak >= n_mv * 2:
                    break  # explored enough clusters; HPWL-improving swaps exhausted

        # ── FD gradient descent (targets congestion/density, blind to HPWL gradient) ──
        # For each of the top-k congested macros: compute ∂proxy/∂x and ∂proxy/∂y via
        # 1-sided FD (probe at pos+δ, compare to best_cost). Apply normalized gradient
        # step, legalize, verify. Repeats until no improvement or step size collapses.
        # Cost: 2*fd_k oracle calls per gradient step + 1 verify ≈ 41 calls/step at k=20.
        oracle_time_est = max(0.5, (time.time() - (deadline - self.sa_time_budget)) / max(1, oracle_calls))
        fd_k = min(n_mv, 20)
        # Run FD if at least 1 full step fits: fd_k * 2 probes + 1 verify + 2 buffer.
        # Threshold is deliberately low — even 1 FD step is worth trying for
        # congestion-dominated benchmarks (ibm06-style) where oracle SA barely helps.
        if deadline - time.time() > oracle_time_est * (fd_k * 2 + 3) and n_mv >= 1:
            _rebuild_state(best_pos, best_ori)
            fd_delta_x = cw * 0.02
            fd_delta_y = ch * 0.02
            # Step size: scale with available room. For dense benchmarks (>75% utilization),
            # use smaller steps since macros are closely packed and large steps cause many
            # legalization passes. For sparse benchmarks, larger steps explore better.
            macro_area_frac = (benchmark.macro_sizes[:n_hard, 0] * benchmark.macro_sizes[:n_hard, 1]).sum().item() / (cw * ch)
            fd_step_size = max(cw, ch) * (0.02 if macro_area_frac > 0.75 else 0.05)
            fd_no_improve = 0; fd_improved = 0; fd_step = 0
            print(f"  [FD] starting, k={fd_k}, step={fd_step_size:.1f}, remaining={deadline-time.time():.0f}s")
            while time.time() < deadline - oracle_time_est * (fd_k * 2 + 4):
                if _cong_weights is not None:
                    top_k = [movable_hard[int(i)] for i in np.argsort(_cong_weights)[-fd_k:][::-1]]
                else:
                    top_k = movable_hard[:fd_k]
                f0 = best_cost
                grad_x = np.zeros(len(top_k))
                grad_y = np.zeros(len(top_k))
                for ii, mi in enumerate(top_k):
                    if time.time() >= deadline - oracle_time_est * ((len(top_k) - ii) * 2 + 4):
                        break
                    ox, oy = float(pos[mi, 0]), float(pos[mi, 1])
                    # x-gradient probe
                    nx = float(np.clip(ox + fd_delta_x, hw[mi], cw - hw[mi]))
                    if abs(nx - ox) > fd_delta_x * 0.1 and not _overlaps(mi, nx, oy):
                        pos[mi, 0] = nx
                        fp = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                        oracle_calls += 1
                        grad_x[ii] = (fp - f0) / fd_delta_x
                        pos[mi, 0] = ox
                    # y-gradient probe
                    ny = float(np.clip(oy + fd_delta_y, hh[mi], ch - hh[mi]))
                    if abs(ny - oy) > fd_delta_y * 0.1 and not _overlaps(mi, ox, ny):
                        pos[mi, 1] = ny
                        fp = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                        oracle_calls += 1
                        grad_y[ii] = (fp - f0) / fd_delta_y
                        pos[mi, 1] = oy
                gnorm = math.sqrt(float(np.sum(grad_x**2 + grad_y**2)))
                if gnorm < 1e-12:
                    break
                old_snap = pos.copy()
                for ii, mi in enumerate(top_k):
                    pos[mi, 0] = float(np.clip(pos[mi, 0] - fd_step_size * grad_x[ii] / gnorm, hw[mi], cw - hw[mi]))
                    pos[mi, 1] = float(np.clip(pos[mi, 1] - fd_step_size * grad_y[ii] / gnorm, hh[mi], ch - hh[mi]))
                pos_t = self._legalize_hard(torch.tensor(pos, dtype=torch.float32), benchmark)
                pos[:] = pos_t.numpy()
                all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
                cost = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                oracle_calls += 1
                if cost < best_cost:
                    best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                    fd_improved += 1; fd_no_improve = 0
                    fd_step_size = min(fd_step_size * 1.2, max(cw, ch) * 0.15)
                    print(f"  [FD] step {fd_step}: {cost:.4f}")
                    for i in movable_hard:
                        for k_ in macro_to_nets[i]:
                            fb_ = _compute_fix_bbox(k_, i)
                            if fb_ is not None:
                                macro_fix_bbox[i][k_] = fb_
                    if _cong_weights is not None:
                        try:
                            _h2 = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                            _v2 = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                            _mc2 = np.array([(_h2+_v2)[c] if 0<=c<len(_h2) else 0.0
                                             for c in [plc.get_grid_cell_of_node(p) for p in _plc_ids]], dtype=np.float32)
                            _thr2 = float(np.percentile(_mc2, 50))
                            _wts2 = np.maximum(0.0, _mc2 - _thr2).astype(np.float64)
                            if _wts2.sum() > 0:
                                _cong_weights = _wts2 / _wts2.sum()
                                _inv_w2 = 1.0 / (_cong_weights + 1e-9)
                                _inv_weights = (_inv_w2 / _inv_w2.sum()).astype(np.float64)
                        except Exception:
                            pass
                else:
                    pos[:] = old_snap
                    all_cx[:n_mac] = pos[:n_mac, 0]; all_cy[:n_mac] = pos[:n_mac, 1]
                    fd_no_improve += 1; fd_step_size *= 0.5
                    if fd_no_improve >= 4 or fd_step_size < max(cw, ch) * 0.002:
                        break
                fd_step += 1
            if fd_step > 0:
                print(f"  [FD] {fd_improved}/{fd_step} steps improved, best={best_cost:.4f}")
            _rebuild_state(best_pos, best_ori)

        # ── Oracle-guided SA fallback (uses remaining time after CD+LNS+FD exit) ──────
        remaining = deadline - time.time()
        oracle_time_est = max(0.5, (time.time() - (deadline - self.sa_time_budget)) / max(1, oracle_calls))
        # Use remaining time as primary control; n_osa_budget is just an upper bound
        # (many iterations are rejected for overlap; time check is the real exit)
        n_osa_budget = max(int(remaining / oracle_time_est) * 5, 100000)
        if remaining > oracle_time_est * 8:
            _rebuild_state(best_pos, best_ori)
            T_start = best_cost * 0.04
            T_end   = best_cost * 0.002
            osa_accepted = 0; osa_improved = 0; osa_step = 0
            t_osa_start = time.time()
            t_osa_end   = deadline - oracle_time_est * 3
            # Adaptive cluster size: start at k_osa_max, reduce if valid rate is too low.
            # Dense benchmarks (ibm09 P_valid=2.8%) need k=1; sparse (ibm01) benefit from k>1.
            # k=6 cluster moves need n_mv >> 240 to justify the overlap rejection overhead.
            # For medium benchmarks (ibm02: 271 macros), k=6 gives only ~5% valid rate → 95%
            # wasted iterations. Use k=1 (single-macro) for all but very large benchmarks.
            # ibm09 (dense, ~270 macros): single-macro P_valid=2.8% → swap_enabled kicks in.
            k_osa_max = 1  # single-macro moves; clusters only cause overhead for ibm-range N
            k_osa_cur = k_osa_max
            _valid_trials = 0; _total_trials = 0  # for adaptive k
            _gauss_trials = 0; _gauss_valid_cnt = 0; _swap_enabled = False
            for osa_step in range(n_osa_budget):
                t_now = time.time()
                if t_now >= t_osa_end:
                    break
                # Time-based temperature: anneal from T_start→T_end over the oracle SA window
                frac = (t_now - t_osa_start) / max(1e-9, t_osa_end - t_osa_start)
                T = T_start * math.exp(math.log(T_end / T_start) * frac)
                scale_osa = max(cw, ch) * (0.10 * (1 - frac) + 0.01)

                # Pick cluster center: 70% congestion-guided, 30% random
                if _cong_weights is not None and random.random() < 0.7:
                    ci_idx = int(np.random.choice(n_mv, p=_cong_weights))
                    ci = movable_hard[ci_idx]
                else:
                    ci = random.choice(movable_hard)
                # Adapt cluster size every 50 trials: reduce k if valid rate too low
                _total_trials += 1
                if _total_trials % 50 == 0 and _total_trials > 0:
                    vrate = _valid_trials / _total_trials
                    # Reduce k when overhead > 10% of oracle time:
                    # overhead_per_oracle ≈ oracle_time_est * 0.1 → P_valid_threshold = 0.003%
                    p_thresh = max(0.0003, 0.001 / oracle_time_est)
                    if vrate < p_thresh and k_osa_cur > 1:
                        k_osa_cur = max(1, k_osa_cur - 1)  # too dense: shrink cluster
                    elif vrate > 0.2 and k_osa_cur < k_osa_max:
                        k_osa_cur = min(k_osa_max, k_osa_cur + 1)  # sparse: grow cluster

                if k_osa_cur > 1:
                    dists_osa = sorted([(abs(pos[j,0]-pos[ci,0])+abs(pos[j,1]-pos[ci,1]), j)
                                        for j in movable_hard if j != ci])
                    cluster = [ci] + [j for _, j in dists_osa[:k_osa_cur - 1]]
                else:
                    cluster = [ci]

                # Swap moves: only for dense benchmarks where Gaussian valid rate < 3%.
                # (ibm09-style: k=6 cluster P_valid≈0, swaps guarantee oracle call rate)
                # Disabled for sparse/moderate benchmarks (ibm02) where Gaussian is better.
                if n_mv >= 2 and _swap_enabled and random.random() < 0.30:
                    # Directed swap: ci = high-cong (already selected above),
                    # j = low-cong (70%) or random (30%) for congestion-reducing swaps
                    if _inv_weights is not None and random.random() < 0.7:
                        j_idx = int(np.random.choice(n_mv, p=_inv_weights))
                        j = movable_hard[j_idx]
                    else:
                        j_idx = random.randrange(n_mv)
                        j = movable_hard[j_idx]
                    if j == ci:
                        j_idx = (j_idx + 1) % n_mv
                        j = movable_hard[j_idx]
                    # Swap positions (always valid since both positions were already legal)
                    old_cxs = [float(pos[ci,0]), float(pos[j,0])]
                    old_cys = [float(pos[ci,1]), float(pos[j,1])]
                    pos[ci,0], pos[ci,1] = old_cxs[1], old_cys[1]
                    pos[j,0],  pos[j,1]  = old_cxs[0], old_cys[0]
                    cluster = [ci, j]
                    _valid_trials += 1
                else:
                    # Save old positions and generate Gaussian candidates
                    old_cxs = [float(pos[i,0]) for i in cluster]
                    old_cys = [float(pos[i,1]) for i in cluster]
                    new_cxs = [float(np.clip(pos[i,0]+random.gauss(0,scale_osa), hw[i], cw-hw[i])) for i in cluster]
                    new_cys = [float(np.clip(pos[i,1]+random.gauss(0,scale_osa), hh[i], ch-hh[i])) for i in cluster]

                    # Apply moves temporarily to pos, then check all cluster overlaps at once
                    for k_idx, i in enumerate(cluster):
                        pos[i,0] = new_cxs[k_idx]; pos[i,1] = new_cys[k_idx]
                    _gauss_trials += 1
                    valid_cluster = all(not _overlaps(i, pos[i,0], pos[i,1]) for i in cluster)
                    if not valid_cluster:
                        for k_idx, i in enumerate(cluster):
                            pos[i,0] = old_cxs[k_idx]; pos[i,1] = old_cys[k_idx]
                        # Dense-mode check: enable swaps when Gaussian valid rate < 3%
                        if _gauss_trials % 100 == 0 and _gauss_trials >= 100:
                            _swap_enabled = (_gauss_valid_cnt / _gauss_trials) < 0.03
                        continue
                    _gauss_valid_cnt += 1
                    _valid_trials += 1

                cost = compute_proxy_cost(
                    torch.tensor(pos, dtype=torch.float32), benchmark, plc
                )["proxy_cost"]
                oracle_calls += 1
                delta = cost - best_cost
                if delta < 0 or (T > 0 and random.random() < math.exp(-delta / T)):
                    osa_accepted += 1
                    if cost < best_cost:
                        best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                        osa_improved += 1
                        print(f"  [oSA] {cost:.4f} (step {osa_step})")
                    # Refresh congestion map every 30 oracle calls (plc is at current pos)
                    if _cong_weights is not None and oracle_calls % 30 == 0:
                        try:
                            _h_cong  = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                            _v_cong  = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                            _total   = _h_cong + _v_cong
                            _cells   = [plc.get_grid_cell_of_node(pid) for pid in _plc_ids]
                            _mc      = np.array([_total[c] if 0 <= c < len(_total) else 0.0 for c in _cells])
                            _thr     = float(np.percentile(_mc, 50))
                            _wts     = np.maximum(0.0, _mc - _thr).astype(np.float64)
                            if _wts.sum() > 0:
                                _cong_weights = _wts / _wts.sum()
                                _inv_wts2 = 1.0 / (_cong_weights + 1e-9)
                                _inv_weights = (_inv_wts2 / _inv_wts2.sum()).astype(np.float64)
                        except Exception:
                            pass
                else:
                    for k_idx, i in enumerate(cluster):
                        pos[i,0] = old_cxs[k_idx]; pos[i,1] = old_cys[k_idx]
            if osa_improved > 0 or osa_accepted > 0:
                print(f"  [oSA] {osa_improved} improved, {osa_accepted}/{osa_step+1} accepted (k={k_osa_cur})")

        # ── Final soft-macro update (revert if no gain) ───────────────────────
        _rebuild_state(best_pos, best_ori)
        pos_with_soft_update = self._update_soft_macros(pos.copy(), benchmark)
        soft_cost = compute_proxy_cost(
            torch.tensor(pos_with_soft_update, dtype=torch.float32), benchmark, plc
        )["proxy_cost"]
        oracle_calls += 1
        if soft_cost < best_cost:
            best_cost = soft_cost
            best_pos = pos_with_soft_update
            print(f"  [soft] update improved: {soft_cost:.4f}")
        else:
            print(f"  [soft] update no gain ({soft_cost:.4f} >= {best_cost:.4f}), kept CT positions")

        print(f"  [CD] final={best_cost:.4f} ({oracle_calls} oracle calls, {pass_num} passes)")
        result = torch.tensor(best_pos, dtype=torch.float32)
        # Skip final legalization if best_pos is already evaluator-valid (all overlaps < 4nm
        # threshold). The final legalization would resolve these micro-overlaps but at the
        # cost of moving macros away from their oracle-verified optimal positions.
        if self._count_hard_overlaps_f32(result, benchmark) > 0:
            return self._legalize_hard(result, benchmark)
        return result


