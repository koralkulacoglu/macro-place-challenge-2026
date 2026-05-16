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
            print("[warn] DREAMPlace not available — using initial positions")
            placement = benchmark.macro_positions.clone()

        # Legalize hard macros; fall back to benchmark initial positions if DREAMPlace
        # left unsolvable overlaps (happens for large dense benchmarks like ibm12)
        placement = self._legalize_hard(placement, benchmark)
        remaining = self._count_hard_overlaps_f32(placement, benchmark)
        if remaining > 0:
            print(f"  [fallback] DREAMPlace left {remaining} overlaps after legalization, using initial positions")
            placement = benchmark.macro_positions.clone()
            placement = self._legalize_hard(placement, benchmark)

        if plc is not None and self.sa_time_budget > 0:
            placement = self._sa_polish(placement, benchmark, plc)

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

            aux_path = write_bookshelf(benchmark, tmpdir)

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
                "random_center_init_flag": 1,
                "ignore_net_degree":  100,
                "num_threads":        8,
                "random_seed":        self.seed,
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

        # Post-DREAMPlace: legalize hard macros to guarantee zero overlaps
        # (DREAMPlace's internal check may pass with tiny near-zero overlaps)
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
        for pass_idx in range(500):
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

        # Report remaining overlaps for debugging
        dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
        dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
        ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
        np.fill_diagonal(ov, False)
        n_ov = int(ov.sum()) // 2
        if n_ov > 0:
            print(f"  [legalize] {pass_idx+1} passes, {n_ov} hard macro overlaps remain")
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

    # ── SA polish stage ───────────────────────────────────────────────────────

    def _sa_polish(
        self,
        placement: torch.Tensor,
        benchmark: Benchmark,
        plc,
    ) -> torch.Tensor:
        """
        Proxy-cost SA polish on hard macros.
        - Soft macros are first repositioned to their centroid (fast, O(nets)).
        - Then SA runs for sa_time_budget seconds using real proxy cost as oracle.
        - 15% swap moves, 85% Gaussian shift moves; large initial move distance.
        """
        n_hard   = benchmark.num_hard_macros
        cw, ch   = benchmark.canvas_width, benchmark.canvas_height
        sizes    = benchmark.macro_sizes.numpy()
        half_w   = sizes[:, 0] / 2
        half_h   = sizes[:, 1] / 2
        movable  = benchmark.get_movable_mask().numpy()
        movable_hard = [i for i in range(n_hard) if movable[i]]
        n_mv = len(movable_hard)

        if not movable_hard:
            return placement

        # Separation matrices for O(N) overlap check — match GAP in _legalize_hard
        _gap = 0.005
        sep_x = (sizes[:n_hard, 0:1] + sizes[:n_hard, 0:1].T) / 2 + _gap
        sep_y = (sizes[:n_hard, 1:2] + sizes[:n_hard, 1:2].T) / 2 + _gap

        def macro_overlaps(idx: int, pos_np: np.ndarray) -> bool:
            dx = np.abs(pos_np[idx, 0] - pos_np[:n_hard, 0])
            dy = np.abs(pos_np[idx, 1] - pos_np[:n_hard, 1])
            mask = (dx < sep_x[idx]) & (dy < sep_y[idx])
            mask[idx] = False
            return bool(mask.any())

        pos = placement.numpy().copy()

        # Soft macro centroid update before SA: move soft macros toward hard macros/ports
        pos = self._update_soft_macros(pos, benchmark)

        # Initial proxy cost (soft macros already updated above)
        init_cost = compute_proxy_cost(
            torch.tensor(pos, dtype=torch.float32), benchmark, plc
        )["proxy_cost"]
        best_pos  = pos.copy()
        best_cost = init_cost
        cur_cost  = init_cost

        print(f"  [SA] starting cost={init_cost:.4f}, budget={self.sa_time_budget}s")

        # Temperature: acceptance is in proxy-cost units; move distance scales with canvas
        T_accept      = 0.05    # allows ~5% worsening at start
        T_accept_end  = 0.0005
        move_scale     = max(cw, ch) * 0.08   # large initial moves (~8% of canvas)
        move_scale_end = max(cw, ch) * 0.005  # small final moves (~0.5%)
        deadline = time.time() + self.sa_time_budget
        step = 0

        while time.time() < deadline:
            frac = max(0.0, (deadline - time.time()) / self.sa_time_budget)
            T_acc  = T_accept_end  + (T_accept  - T_accept_end)  * frac
            shift  = move_scale_end + (move_scale - move_scale_end) * frac

            # 15% swap moves, 85% Gaussian shift
            if n_mv >= 2 and random.random() < 0.15:
                i = random.choice(movable_hard)
                j = random.choice(movable_hard)
                if i == j:
                    step += 1
                    continue
                old_i, old_j = pos[i].copy(), pos[j].copy()
                pos[i] = old_j; pos[j] = old_i
                if macro_overlaps(i, pos) or macro_overlaps(j, pos):
                    pos[i] = old_i; pos[j] = old_j
                    step += 1
                    continue
            else:
                i = movable_hard[step % n_mv] if step % 5 != 0 else random.choice(movable_hard)
                old = pos[i].copy()
                pos[i, 0] = np.clip(pos[i, 0] + random.gauss(0, shift), half_w[i], cw - half_w[i])
                pos[i, 1] = np.clip(pos[i, 1] + random.gauss(0, shift), half_h[i], ch - half_h[i])
                if macro_overlaps(i, pos):
                    pos[i] = old
                    step += 1
                    continue
                old_i, old_j = old, None  # track for revert on reject

            # Update soft macros to track new hard macro positions before evaluating cost
            self._update_soft_macros(pos, benchmark)

            new_cost = compute_proxy_cost(
                torch.tensor(pos, dtype=torch.float32), benchmark, plc
            )["proxy_cost"]
            delta = new_cost - cur_cost  # compare to current position, not global best
            if new_cost < best_cost:
                best_cost = new_cost
                best_pos  = pos.copy()
            if delta < 0 or random.random() < math.exp(-delta / max(T_acc, 1e-10)):
                cur_cost = new_cost  # accept
            else:
                # Reject: revert hard macro(s) and soft macros
                pos[i] = old_i
                if old_j is not None:
                    pos[j] = old_j
                self._update_soft_macros(pos, benchmark)

            step += 1

        print(f"  [SA] {step} steps, final best={best_cost:.4f}")
        result = torch.tensor(best_pos, dtype=torch.float32)
        # Final legalization pass to guarantee zero overlaps (SA's gap tolerance may differ)
        return self._legalize_hard(result, benchmark)
