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

# Stub out CUDA ops that were disabled at build time (CUDA 12.4 CUB API incompatibility).
# Must run before any DREAMPlace module imports so module-level references resolve correctly.
# CPU fallbacks are provided where available (pin_pos_cpp for pin_pos_cuda).
if any(p in sys.path for p in [_DP_INSTALL, _DP_PKG]):
    import numpy as _np_compat
    if not hasattr(_np_compat, 'string_'):
        _np_compat.string_ = _np_compat.bytes_  # NumPy 2.0 compat
    from types import ModuleType as _ModuleType
    _dp_cuda_stubs = {
        'dreamplace.ops.pin_pos.pin_pos_cuda': 'dreamplace.ops.pin_pos.pin_pos_cpp',
        'dreamplace.ops.pin_pos.pin_pos_cuda_segment': 'dreamplace.ops.pin_pos.pin_pos_cpp',
        'dreamplace.ops.k_reorder.k_reorder_cuda': None,
        'dreamplace.ops.global_swap.global_swap_cuda': None,
        'dreamplace.ops.independent_set_matching.independent_set_matching_cuda': None,
    }
    for _cud, _cpu in _dp_cuda_stubs.items():
        if _cud not in sys.modules:
            try:
                __import__(_cud)
            except (ImportError, Exception):
                _stub = _ModuleType(_cud.split('.')[-1])
                if _cpu:
                    try:
                        _cpu_mod = __import__(_cpu, fromlist=['forward'])
                        for _attr in dir(_cpu_mod):
                            if not _attr.startswith('__'):
                                setattr(_stub, _attr, getattr(_cpu_mod, _attr))
                    except (ImportError, Exception):
                        pass
                sys.modules[_cud] = _stub

# Suppress DREAMPlace's verbose logging
logging.getLogger().setLevel(logging.WARNING)
# Unbuffered output for live progress in nohup/pipe contexts; UTF-8 for Windows cp1252 compat
sys.stdout.reconfigure(line_buffering=True, encoding="utf-8", errors="replace")

from macro_place.benchmark import Benchmark
from macro_place.objective  import compute_proxy_cost, compute_overlap_metrics

# Import bookshelf writer from same directory
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))
from bookshelf import write_bookshelf, dreamplace_nodes_to_tensor


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_plc(benchmark: Benchmark):
    """Reload PlacementCost for a benchmark — needed to call compute_proxy_cost."""
    try:
        from macro_place.loader import load_benchmark

        # Use forward-slash strings so plc_client_os.rsplit('/') works on Windows too
        base_ibm = "external/MacroPlacement/Testcases/ICCAD04/" + benchmark.name
        netlist = base_ibm + "/netlist.pb.txt"
        plc_file = base_ibm + "/initial.plc"
        if Path(netlist).exists():
            _, plc = load_benchmark(
                netlist,
                plc_file if Path(plc_file).exists() else None,
            )
            return plc

        # NG45 paths
        ng45_map = {
            "ariane133": "ariane133", "ariane136": "ariane136",
            "nvdla": "nvdla",         "mempool_tile": "mempool_tile",
        }
        ng45_name = ng45_map.get(benchmark.name)
        if ng45_name:
            base_ng = ("external/MacroPlacement/Flows/NanGate45/"
                       + ng45_name + "/netlist/output_CT_Grouping")
            ng_netlist = base_ng + "/netlist.pb.txt"
            ng_plc = base_ng + "/initial.plc"
            if Path(ng_netlist).exists():
                _, plc = load_benchmark(
                    ng_netlist,
                    ng_plc if Path(ng_plc).exists() else None,
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


def _polish_worker(args):
    """Module-level worker for parallel SA (fork context on Linux).

    Reloads benchmark + plc from disk so each worker has its own independent
    PlacementCost state. Returns (placement_np, proxy_cost).
    """
    bench_name, placement_np, sa_budget, seed = args
    try:
        from macro_place.loader import load_benchmark
        base_ibm = "external/MacroPlacement/Testcases/ICCAD04/" + bench_name
        netlist   = base_ibm + "/netlist.pb.txt"
        plc_file  = base_ibm + "/initial.plc"
        if not Path(netlist).exists():
            ng45_map = {
                "ariane133": "ariane133", "ariane136": "ariane136",
                "nvdla": "nvdla", "mempool_tile": "mempool_tile",
            }
            ng45_name = ng45_map.get(bench_name)
            if ng45_name:
                base_ng  = ("external/MacroPlacement/Flows/NanGate45/"
                            + ng45_name + "/netlist/output_CT_Grouping")
                netlist  = base_ng + "/netlist.pb.txt"
                plc_file = base_ng + "/initial.plc"
        bench, plc = load_benchmark(
            netlist, plc_file if Path(plc_file).exists() else None
        )
        placement = torch.from_numpy(placement_np).clone()
        placer    = KoralPlacer(sa_time_budget=sa_budget, seed=seed)
        result    = placer._cd_lns_polish(placement, bench, plc)
        cost      = compute_proxy_cost(result, bench, plc)["proxy_cost"]
        print(f"  [worker-s{seed}] done, cost={cost:.4f}", flush=True)
        return result.numpy(), cost
    except Exception as e:
        import traceback
        print(f"  [worker-s{seed}] ERROR: {e}\n{traceback.format_exc()}", flush=True)
        return placement_np, float('inf')


# ── Main placer ───────────────────────────────────────────────────────────────

class KoralPlacer:
    def __init__(
        self,
        target_density: float = 0.0,     # 0 = auto from utilization
        density_weight: float = 8e-5,
        gamma: float = 4.0,
        sa_time_budget: int = int(os.environ.get("KORAL_SA_BUDGET", "3480")),  # 1hr judge limit
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
        # Full legalization for significant (>4nm) overlaps; micro-legalize for sub-4nm.
        # CT positions for ibm06 have 47 sub-4nm float64 overlaps. SA preserves these
        # because SA only rejects NEW candidate overlaps, never fixes existing ones.
        # micro_legalize here resolves CT overlaps before SA sees them (CT macros are
        # grid-placed, not at boundaries, so micro_legalize converges in 1-2 passes).
        if self._count_significant_overlaps(ct, benchmark, threshold=0.004) > 0:
            ct_legal = self._legalize_hard(ct, benchmark)
        else:
            ct_legal = ct
            # Note: pre-SA micro_legalize causes cascade amplification on dense benchmarks
            # (ibm06: 47 pairs chain-react, each 7nm push creates new overlaps → 15μm max
            # displacement → oracle 1.87 vs CT's 1.66). SA preserves existing sub-4nm
            # overlaps but also won't place macros INTO overlapping positions (via _overlaps
            # filter). Post-SA micro_legalize handles any residual overlaps.

        placement = ct_legal  # default: CT positions

        n_mv_hard = sum(1 for i in range(benchmark.num_hard_macros)
                        if not benchmark.macro_fixed[i])

        # Skip DREAMPlace if SA budget is too short (DREAMPlace would eat all the time).
        # CPU DREAMPlace: ~2-3 min. Only worth running if SA gets meaningful time after.
        if _dreamplace_available() and self.sa_time_budget >= 300:
            # CT-init (warm start): better for large benchmarks (ibm02-18) where center-init diverges.
            # Center-init (random scatter): historically best for ibm01 (finds 0.9221 vs CT 1.04).
            # Strategy: always try CT-init; also try center-init for small benchmarks (≤260 macros).
            ct_cost = compute_proxy_cost(ct_legal, benchmark, plc)["proxy_cost"] if plc else float('inf')
            best_dp_cost = ct_cost
            best_placement = ct_legal
            _n_mv_dp = sum(1 for i in range(benchmark.num_hard_macros) if not benchmark.macro_fixed[i])

            # ibm01 (246 macros): center-init finds global min (0.9221 vs CT 1.04); CT-init diverges.
            # ibm02+ (259+ macros): CT-init warm start; center-init always diverges for these.
            # Threshold 248 cleanly separates ibm01 (246) from ibm02 (259).
            for _dp_center_init in ([True] if _n_mv_dp <= 248 else [False]):
                _label = "center-init" if _dp_center_init else "CT-init"
                try:
                    dp = self._run_dreamplace(benchmark, center_init=_dp_center_init,
                                              fix_soft=False, dp_seed=self.seed)
                    dp = self._legalize_hard(dp, benchmark)
                    if self._count_hard_overlaps_f32(dp, benchmark) == 0 and plc is not None:
                        dp_cost = compute_proxy_cost(dp, benchmark, plc)["proxy_cost"]
                        if dp_cost < best_dp_cost:
                            print(f"  [start] DREAMPlace({_label}) {dp_cost:.4f} < best {best_dp_cost:.4f}")
                            best_dp_cost = dp_cost
                            best_placement = dp
                        else:
                            print(f"  [start] DREAMPlace({_label}) {dp_cost:.4f} >= best {best_dp_cost:.4f}")
                except Exception as e:
                    print(f"  [start] DREAMPlace({_label}) failed: {e}")

            placement = best_placement

        if plc is not None and self.sa_time_budget > 0:
            # Parallel SA: spawn N independent workers with different seeds, keep best result.
            # Uses fork (Linux/Docker only) so workers inherit parent memory without pickling.
            # Falls back to sequential on Windows (spawn has import-path complexity).
            n_workers = (min(4, os.cpu_count() or 1)
                         if sys.platform != 'win32' else 1)
            if n_workers > 1:
                import multiprocessing as _mp
                print(f"  [parallel-SA] {n_workers} workers, seeds {self.seed}..{self.seed+n_workers-1}")
                # Worker 0: exact best placement (exploitation).
                # Workers 1+: increasingly perturbed starts (exploration diversity).
                # Perturbation ensures workers explore different basins of attraction.
                _base_np = placement.numpy()
                _mv_mask = benchmark.get_movable_mask().numpy().astype(bool)
                _hw_arr = benchmark.macro_sizes[:, 0].numpy() / 2
                _hh_arr = benchmark.macro_sizes[:, 1].numpy() / 2
                _cw_f, _ch_f = float(benchmark.canvas_width), float(benchmark.canvas_height)
                _rng_p = np.random.RandomState(self.seed)
                args_list = []
                for i in range(n_workers):
                    _start = _base_np.copy()
                    if i > 0:
                        _sigma = max(_cw_f, _ch_f) * (0.015 * i)  # 1.5%, 3%, 4.5% for workers 1,2,3
                        for mi in range(benchmark.num_hard_macros):
                            if _mv_mask[mi]:
                                _start[mi, 0] = float(np.clip(
                                    _start[mi, 0] + _rng_p.normal(0, _sigma),
                                    _hw_arr[mi], _cw_f - _hw_arr[mi]))
                                _start[mi, 1] = float(np.clip(
                                    _start[mi, 1] + _rng_p.normal(0, _sigma),
                                    _hh_arr[mi], _ch_f - _hh_arr[mi]))
                    args_list.append((benchmark.name, _start, self.sa_time_budget, self.seed + i))
                ctx = _mp.get_context('fork')
                result_q = ctx.Queue()

                def _worker_fn(args):
                    try:
                        res = _polish_worker(args)
                        result_q.put(res)
                    except Exception as e:
                        print(f"  [worker] error: {e}", flush=True)
                        result_q.put((args[1], float('inf')))

                procs = [ctx.Process(target=_worker_fn, args=(a,)) for a in args_list]
                for p in procs: p.start()
                for p in procs: p.join()
                results = [result_q.get() for _ in procs]

                best_cost, best_np = float('inf'), None
                for res_np, cost in results:
                    if cost < best_cost:
                        best_cost, best_np = cost, res_np
                if best_np is not None:
                    placement = torch.from_numpy(best_np)
                print(f"  [parallel-SA] best cost {best_cost:.4f}")
            else:
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
            # Headroom: center-init starts from scatter → needs 20% headroom to allow spreading.
            # CT-init starts near-optimal → 5% headroom prevents over-constraining the layout.
            target_density = min(0.95, utilization + (0.20 if center_init else 0.05))

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

            # Adaptive DREAMPlace iterations based on GPU availability.
            # GPU: use CUDA when available. pin_pos_cuda compiles fine with CUDA 12.4;
            # only pin_pos_cuda_segment + 3 detailed-place ops are disabled (CUB API compat).
            # detailed_place_flag=0 so those disabled ops are never invoked.
            n_mv_hard = len(movable_hard)
            # Use GPU only if CUDA is available AND the real pin_pos_cuda .so loaded.
            # If pin_pos_cuda failed to import, our stub substitutes pin_pos_cpp (CPU).
            # Calling a CPU op with GPU tensors (gpu=1 mode) causes a runtime crash.
            _gpu = 0
            if torch.cuda.is_available():
                try:
                    import dreamplace.ops.pin_pos.pin_pos_cuda as _ppc
                    if getattr(_ppc, '__file__', None) is not None:
                        _gpu = 1
                except Exception:
                    pass
            if _gpu:
                # GPU: 500+1000 validated iteration counts. More iterations (2000+3000)
                # cause DREAMPlace to aggressively push WL at the cost of macro separation,
                # resulting in 140-200 hard macro overlaps that legalization can't fix.
                # 500+1000 gives good WL improvement with manageable overlap count (0-5).
                _iters1, _iters2 = 500, 1000
            else:
                # CPU: limit to avoid timeout.
                _iters1, _iters2 = 200, 300

            # Build DREAMPlace params
            params_dict = {
                "aux_input":          aux_path,
                "gpu":                _gpu,
                "target_density":     target_density,
                "density_weight":     self.density_weight,
                "gamma":              self.gamma,
                "macro_place_flag":   0,  # skip Hannan grid; global placement spreads macros first
                "legalize_flag":        0,
                "abacus_legalize_flag": 0,
                "detailed_place_flag":  0,
                "global_place_flag":  1,
                "enable_fillers":     1,
                "stop_overflow":      0.001 if center_init else 0.03,
                # center-init: 0.001 keeps rollback window at [0.11%, 0.4%] which center-init
                # never reaches, effectively disabling the divergence rollback. DREAMPlace runs
                # all iterations without early stopping. 0.07 and 0.01 both triggered rollback
                # at different overflow ranges (7-28% and 1-4% respectively).
                # CT-init: 3% — CT positions start at ~4% overflow; DREAMPlace refines slightly.
                "gp_noise_ratio":     0.025 if center_init else 0.01,
                "random_center_init_flag": 1 if center_init else 0,
                "ignore_net_degree":  100,
                "num_threads":        8,
                "random_seed":        dp_seed if dp_seed is not None else self.seed,
                "scale_factor":       0.0,
                "result_dir":         os.path.join(tmpdir, "results"),
                "global_place_stages": [
                    {
                        "num_bins_x": 128, "num_bins_y": 128,  # coarse first
                        "iteration": _iters1, "learning_rate": 0.01,
                        "wirelength": "weighted_average",
                        "optimizer": "nesterov",
                        "Llambda_density_weight_iteration": 1,
                        "Lsub_iteration": 1,
                    },
                    {
                        "num_bins_x": 512, "num_bins_y": 512,  # fine second
                        "iteration": _iters2, "learning_rate": 0.01,
                        "wirelength": "weighted_average",
                        "optimizer": "nesterov",
                        "Llambda_density_weight_iteration": 1,
                        "Lsub_iteration": 1,
                    },
                ],
                "macro_halo_x": 50,
                "macro_halo_y": 50,
                "plot_flag":    0,
                "dtype":        "float32",
            }

            params_path = os.path.join(tmpdir, "params.json")
            with open(params_path, "w") as f:
                json.dump(params_dict, f)

            params = Params.Params(); params.load(params_path)
            placedb = PlaceDB.PlaceDB(); placedb(params)

            # Compatibility fix: newer DREAMPlace sets quad_penalty=True during a convergence
            # stall. If init_density was set by a fence-region path before obj_fn's lazy init
            # block runs, quad_penalty_coeff is never initialized. Patch PlaceObj.obj_fn so
            # the guard runs before the attribute access.
            # NonLinearPlace imports PlaceObj as a bare name; patch that module to affect the
            # same class object (not dreamplace.PlaceObj which is a different sys.modules key).
            try:
                import PlaceObj as _po_mod
            except ImportError:
                try:
                    import dreamplace.PlaceObj as _po_mod
                except ImportError:
                    _po_mod = None
            if _po_mod is not None:
                _PlaceObj_cls = _po_mod.PlaceObj
                if not getattr(_PlaceObj_cls, '_koral_qpc_patched', False):
                    _orig_obj_fn = _PlaceObj_cls.obj_fn
                    def _patched_obj_fn(self, pos, _orig=_orig_obj_fn):
                        if not hasattr(self, 'quad_penalty_coeff') and getattr(self, 'quad_penalty', False):
                            self.quad_penalty_coeff = 0.0  # disables quadratic penalty safely
                        return _orig(self, pos)
                    _PlaceObj_cls.obj_fn = _patched_obj_fn
                    _PlaceObj_cls._koral_qpc_patched = True

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

    @staticmethod
    def _count_significant_overlaps(placement: torch.Tensor, benchmark: Benchmark,
                                     threshold: float = 0.004,
                                     skip_fixed_fixed: bool = False) -> int:
        """Count hard macro pairs with overlap depth > threshold microns.
        TILOS ignores sub-4nm overlaps (float32 precision); use threshold=0.004 to match.
        skip_fixed_fixed=True: ignore fixed-fixed overlaps that no legalization can fix."""
        n = benchmark.num_hard_macros
        pos = placement[:n].numpy().astype(np.float32)
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float32)
        sep_x = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        dx = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
        dy = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
        ov = ((sep_x - dx) > threshold) & ((sep_y - dy) > threshold)
        np.fill_diagonal(ov, False)
        if skip_fixed_fixed:
            movable = benchmark.get_movable_mask()[:n].numpy().astype(bool)
            at_least_one_movable = movable[:, np.newaxis] | movable[np.newaxis, :]
            ov = ov & at_least_one_movable
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

    def _micro_legalize(self, placement: torch.Tensor, benchmark: Benchmark) -> torch.Tensor:
        """
        Minimal-displacement legalization to eliminate ALL physical overlaps.
        Uses 3nm gap (vs 5nm in _legalize_hard) to minimize cost impact.
        Called before SA (to fix CT overlaps) and after SA (to fix any residual).
        """
        GAP = 0.003  # 3nm — survives float32 conversion
        n = benchmark.num_hard_macros
        sizes = benchmark.macro_sizes[:n].numpy().astype(np.float64)
        movable = benchmark.get_movable_mask()[:n].numpy().astype(bool)
        cw = float(benchmark.canvas_width)
        ch = float(benchmark.canvas_height)
        half_w = sizes[:, 0] / 2
        half_h = sizes[:, 1] / 2
        pos = placement[:n].numpy().copy().astype(np.float64)
        sep_x_exact = (sizes[:, 0:1] + sizes[:, 0:1].T) / 2
        sep_y_exact = (sizes[:, 1:2] + sizes[:, 1:2].T) / 2
        sep_x = sep_x_exact + GAP
        sep_y = sep_y_exact + GAP

        for _ in range(2000):
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break
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
                    old_i, old_j = pos[i].copy(), pos[j].copy()
                    # Resolve in minimum-overlap direction first.
                    # Only try the other direction if: (a) primary fails AND (b) other
                    # direction requires < 10nm movement. This fixes stuck pairs (boundary-
                    # clamped macros with tiny overlaps in both dims) without triggering
                    # large movements for ibm06-type pairs (small x-overlap, large y-extent).
                    # Primary direction (minimum overlap)
                    _TINY = 0.100  # 100nm: max secondary movement to avoid cascade
                    if ox <= oy:
                        need = ox; sign = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                        if both:
                            xi_n = np.clip(pos[i, 0] + sign * need / 2, half_w[i], cw - half_w[i])
                            xj_n = np.clip(pos[j, 0] - sign * need / 2, half_w[j], cw - half_w[j])
                            got_i = abs(xi_n - pos[i, 0]); got_j = abs(xj_n - pos[j, 0])
                            if got_i + got_j < need - 1e-9:
                                xi_n = np.clip(pos[i, 0] + sign * (need - got_j), half_w[i], cw - half_w[i])
                                xj_n = np.clip(pos[j, 0] - sign * (need - got_i), half_w[j], cw - half_w[j])
                            pos[i, 0] = xi_n; pos[j, 0] = xj_n
                        elif movable[i]:
                            pos[i, 0] = np.clip(pos[i, 0] + sign * need, half_w[i], cw - half_w[i])
                        else:
                            pos[j, 0] = np.clip(pos[j, 0] - sign * need, half_w[j], cw - half_w[j])
                        # If still overlapping after primary (clamped in x), try y
                        # ONLY if the required y movement is tiny (< 10nm) to avoid cascade.
                        dx2 = abs(pos[i, 0] - pos[j, 0]); dy2 = abs(pos[i, 1] - pos[j, 1])
                        if (sep_x_exact[i, j] - dx2 > 0) and (sep_y_exact[i, j] - dy2 > 0):
                            oy2 = sep_y[i, j] - dy2
                            if 0 < oy2 <= _TINY:
                                sy = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                                if both:
                                    pos[i, 1] = np.clip(pos[i, 1] + sy * oy2 / 2, half_h[i], ch - half_h[i])
                                    pos[j, 1] = np.clip(pos[j, 1] - sy * oy2 / 2, half_h[j], ch - half_h[j])
                                elif movable[i]:
                                    pos[i, 1] = np.clip(pos[i, 1] + sy * oy2, half_h[i], ch - half_h[i])
                                else:
                                    pos[j, 1] = np.clip(pos[j, 1] - sy * oy2, half_h[j], ch - half_h[j])
                    else:
                        need = oy; sign = 1.0 if pos[i, 1] >= pos[j, 1] else -1.0
                        if both:
                            yi_n = np.clip(pos[i, 1] + sign * need / 2, half_h[i], ch - half_h[i])
                            yj_n = np.clip(pos[j, 1] - sign * need / 2, half_h[j], ch - half_h[j])
                            got_i = abs(yi_n - pos[i, 1]); got_j = abs(yj_n - pos[j, 1])
                            if got_i + got_j < need - 1e-9:
                                yi_n = np.clip(pos[i, 1] + sign * (need - got_j), half_h[i], ch - half_h[i])
                                yj_n = np.clip(pos[j, 1] - sign * (need - got_i), half_h[j], ch - half_h[j])
                            pos[i, 1] = yi_n; pos[j, 1] = yj_n
                        elif movable[i]:
                            pos[i, 1] = np.clip(pos[i, 1] + sign * need, half_h[i], ch - half_h[i])
                        else:
                            pos[j, 1] = np.clip(pos[j, 1] - sign * need, half_h[j], ch - half_h[j])
                        # If still overlapping after primary (clamped in y), try x if tiny
                        dx2 = abs(pos[i, 0] - pos[j, 0]); dy2 = abs(pos[i, 1] - pos[j, 1])
                        if (sep_x_exact[i, j] - dx2 > 0) and (sep_y_exact[i, j] - dy2 > 0):
                            ox2 = sep_x[i, j] - dx2
                            if 0 < ox2 <= _TINY:
                                sx = 1.0 if pos[i, 0] >= pos[j, 0] else -1.0
                                if both:
                                    pos[i, 0] = np.clip(pos[i, 0] + sx * ox2 / 2, half_w[i], cw - half_w[i])
                                    pos[j, 0] = np.clip(pos[j, 0] - sx * ox2 / 2, half_w[j], cw - half_w[j])
                                elif movable[i]:
                                    pos[i, 0] = np.clip(pos[i, 0] + sx * ox2, half_w[i], cw - half_w[i])
                                else:
                                    pos[j, 0] = np.clip(pos[j, 0] - sx * ox2, half_w[j], cw - half_w[j])
                    if not (np.allclose(pos[i], old_i) and np.allclose(pos[j], old_j)):
                        changed = True
            if not changed:
                break

        # Diagonal push for any remaining cyclic stuck pairs (1000 passes)
        for _ in range(1000):
            dx_mat = np.abs(pos[:, 0:1] - pos[np.newaxis, :, 0])
            dy_mat = np.abs(pos[:, 1:2] - pos[np.newaxis, :, 1])
            is_ov = (sep_x_exact - dx_mat > 0) & (sep_y_exact - dy_mat > 0)
            np.fill_diagonal(is_ov, False)
            if not is_ov.any():
                break
            for i, j in zip(*np.where(is_ov)):
                if i >= j or (not movable[i] and not movable[j]):
                    continue
                dx_v = pos[i, 0] - pos[j, 0]; dy_v = pos[i, 1] - pos[j, 1]
                d = math.sqrt(dx_v * dx_v + dy_v * dy_v)
                fx, fy = (1.0, 0.0) if d < 1e-9 else (dx_v / d, dy_v / d)
                need_x = sep_x[i, j] - abs(dx_v)
                need_y = sep_y[i, j] - abs(dy_v)
                s = max(need_x, need_y) * 0.5
                if movable[i]:
                    pos[i, 0] = np.clip(pos[i, 0] + fx * s, half_w[i], cw - half_w[i])
                    pos[i, 1] = np.clip(pos[i, 1] + fy * s, half_h[i], ch - half_h[i])
                if movable[j]:
                    pos[j, 0] = np.clip(pos[j, 0] - fx * s, half_w[j], cw - half_w[j])
                    pos[j, 1] = np.clip(pos[j, 1] - fy * s, half_h[j], ch - half_h[j])

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
            n_cands = 50

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

        lns_k = 12
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
        # Run FD for ALL benchmarks. For congestion-dominated (wl_frac < 4%), use larger
        # probe distance (5% canvas) so position changes are large enough to perturb
        # congestion grid bins and produce a non-zero gradient signal.
        if deadline - time.time() > oracle_time_est * (fd_k * 2 + 3) and n_mv >= 1:
            _rebuild_state(best_pos, best_ori)
            _fd_probe = 0.05 if _wl_frac < 0.04 else 0.02
            fd_delta_x = cw * _fd_probe
            fd_delta_y = ch * _fd_probe
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

        # ── HPWL-guided SA with periodic oracle checkpoints ──────────────────────
        # Uses HPWL as fast surrogate (O(degree), ~0.01ms/call) for SA acceptance,
        # with proxy oracle checkpoint every HPWL_CKPT_N accepted moves (~0.5s/call).
        # Explores ~50× more positions per oracle call budget vs pure oracle SA.
        # Move mix: 65% Gaussian translation, 15% rotation, 20% swap (dense only).
        # Adaptive HPWL_CKPT_N: frequent oracle for congestion-dominated benchmarks.
        remaining = deadline - time.time()
        oracle_time_est = max(0.5, (time.time() - (deadline - self.sa_time_budget)) / max(1, oracle_calls))
        # More frequent oracle for congestion-dominated benchmarks (HPWL ≠ proxy there).
        HPWL_CKPT_N = max(5, min(50, int(50 * _wl_frac * 1.5)))

        if remaining > oracle_time_est * 8:
            _rebuild_state(best_pos, best_ori)

            # Calibrate HPWL temperature from distribution of |delta_hpwl| on random valid moves
            sample_deltas = []
            for _ in range(min(400, n_mv * 20)):
                i = random.choice(movable_hard)
                cx = float(np.clip(pos[i, 0] + random.gauss(0, max(cw, ch) * 0.08), hw[i], cw - hw[i]))
                cy = float(np.clip(pos[i, 1] + random.gauss(0, max(cw, ch) * 0.08), hh[i], ch - hh[i]))
                if not _overlaps(i, cx, cy):
                    d = abs(_delta_hpwl(i, cx, cy, int(ori[i])))
                    if d > 0:
                        sample_deltas.append(d)

            if len(sample_deltas) >= 10:
                hpwl_T_start = float(np.percentile(sample_deltas, 60)) * 2.0
                hpwl_T_end   = float(np.percentile(sample_deltas,  5)) * 0.05
            else:
                hpwl_T_start = (cw + ch) * 0.3
                hpwl_T_end   = (cw + ch) * 0.003
            print(f"  [hSA] T={hpwl_T_start:.2f}->{hpwl_T_end:.4f}, ckpt={HPWL_CKPT_N}, remaining={remaining:.0f}s")

            accepted_since_ckpt = 0
            hsa_improved = 0; hsa_accepted = 0; hsa_step = 0; hsa_oracle = 0
            _hsa_gauss_tried = 0; _hsa_gauss_valid = 0; _hsa_swap_enabled = False
            t_hsa_start = time.time()
            t_hsa_end   = deadline - oracle_time_est * 2
            # Reheating state: if no oracle improvement in K calls, reheat temperature
            _reheat_no_improve = 0; _reheat_count = 0; _reheat_K = 20; _reheat_max = 4
            T_hsa_current_start = hpwl_T_start  # tracks current cycle start temp for reheat

            while time.time() < t_hsa_end:
                frac = (time.time() - t_hsa_start) / max(1e-9, t_hsa_end - t_hsa_start)
                T_hsa = T_hsa_current_start * math.exp(
                    math.log(max(hpwl_T_end, T_hsa_current_start * 1e-6) / T_hsa_current_start) * frac
                )
                # Multi-phase sigma: broad exploration → focused exploitation → fine-tuning.
                # Resets after each reheat since frac restarts from 0.
                if frac < 0.15:
                    scale_hsa = max(cw, ch) * 0.12
                elif frac < 0.75:
                    scale_hsa = max(cw, ch) * 0.05
                else:
                    scale_hsa = max(cw, ch) * 0.012

                # Pick macro: 70% congestion-guided, 30% random
                if _cong_weights is not None and random.random() < 0.7:
                    ci = movable_hard[int(np.random.choice(n_mv, p=_cong_weights))]
                else:
                    ci = random.choice(movable_hard)

                hsa_step += 1
                r = random.random()

                if r < 0.15:
                    # Rotation move: try next orientation (no translation)
                    new_o = (int(ori[ci]) + 1) % 4
                    delta_h = _delta_hpwl(ci, pos[ci, 0], pos[ci, 1], new_o)
                    if delta_h < 0 or (T_hsa > 0 and random.random() < math.exp(
                            -max(delta_h, 0.0) / T_hsa)):
                        _accept_move(ci, pos[ci, 0], pos[ci, 1], new_o)
                        hsa_accepted += 1; accepted_since_ckpt += 1

                elif r < 0.35 and _hsa_swap_enabled and n_mv >= 2:
                    # Swap move: useful for dense benchmarks where Gaussian has <3% valid rate
                    j_idx = random.randrange(n_mv); j = movable_hard[j_idx]
                    while j == ci: j_idx = (j_idx + 1) % n_mv; j = movable_hard[j_idx]
                    old_ci = (pos[ci, 0], pos[ci, 1]); old_cj = (pos[j, 0], pos[j, 1])
                    if not _overlaps_excl(ci, old_cj[0], old_cj[1], j) and \
                       not _overlaps_excl(j, old_ci[0], old_ci[1], ci):
                        delta_h = _swap_delta_hpwl(ci, j)
                        if delta_h < 0 or (T_hsa > 0 and random.random() < math.exp(
                                -max(delta_h, 0.0) / T_hsa)):
                            _accept_move(ci, old_cj[0], old_cj[1], int(ori[ci]))
                            _accept_move(j, old_ci[0], old_ci[1], int(ori[j]))
                            hsa_accepted += 1; accepted_since_ckpt += 1

                else:
                    # Gaussian translation move (primary)
                    cx = float(np.clip(pos[ci, 0] + random.gauss(0, scale_hsa), hw[ci], cw - hw[ci]))
                    cy = float(np.clip(pos[ci, 1] + random.gauss(0, scale_hsa), hh[ci], ch - hh[ci]))
                    _hsa_gauss_tried += 1
                    if _overlaps(ci, cx, cy):
                        # Track validity rate; enable swaps if Gaussian valid rate < 3%
                        if _hsa_gauss_tried % 100 == 0:
                            _hsa_swap_enabled = (_hsa_gauss_valid / _hsa_gauss_tried) < 0.03
                        continue
                    _hsa_gauss_valid += 1
                    delta_h = _delta_hpwl(ci, cx, cy, int(ori[ci]))
                    if delta_h < 0 or (T_hsa > 0 and random.random() < math.exp(
                            -max(delta_h, 0.0) / T_hsa)):
                        _accept_move(ci, cx, cy, int(ori[ci]))
                        hsa_accepted += 1; accepted_since_ckpt += 1

                if accepted_since_ckpt >= HPWL_CKPT_N:
                    cost = compute_proxy_cost(
                        torch.tensor(pos, dtype=torch.float32), benchmark, plc
                    )["proxy_cost"]
                    oracle_calls += 1; hsa_oracle += 1
                    # Tighter revert for congestion-dominated: HPWL misleads SA far from oracle-optimal.
                    _revert_thr = 1.03 if _wl_frac < 0.05 else 1.10
                    if cost < best_cost:
                        best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                        hsa_improved += 1; _reheat_no_improve = 0
                        print(f"  [hSA] {cost:.4f}")
                    else:
                        _reheat_no_improve += 1
                        if cost > best_cost * _revert_thr:
                            _rebuild_state(best_pos, best_ori)
                    # Reheat on stagnation: if no improvement in _reheat_K oracle calls,
                    # reset temperature to escape local minima.
                    # Last reheat: add perturbation (displace ~5% of macros) for deeper escape.
                    if (_reheat_count >= _reheat_max and _reheat_no_improve >= _reheat_K
                            and deadline - time.time() > oracle_time_est * 30):
                        # All reheats exhausted + stagnating; exit hSA early so oracle SA tail gets time.
                        # Only exit if >30 oracle calls of time remain (worth running oSA).
                        break
                    if (_reheat_no_improve >= _reheat_K and _reheat_count < _reheat_max
                            and time.time() < t_hsa_end - oracle_time_est * 20):
                        if _reheat_count < _reheat_max - 1:
                            # Standard reheat: restore best and raise temperature
                            T_hsa_current_start = hpwl_T_start * 0.4
                            t_hsa_start = time.time()
                            _rebuild_state(best_pos, best_ori)
                        else:
                            # Final reheat: perturb a handful of macros for deeper exploration
                            _perturb = best_pos.copy(); _perturb_ori = best_ori.copy()
                            _n_perturb = max(2, n_mv // 15)
                            _pscale = max(cw, ch) * 0.18
                            _pmacros = random.sample(movable_hard, min(_n_perturb, n_mv))
                            for _pmi in _pmacros:
                                _pcx = float(np.clip(_perturb[_pmi, 0] + random.gauss(0, _pscale), hw[_pmi], cw - hw[_pmi]))
                                _pcy = float(np.clip(_perturb[_pmi, 1] + random.gauss(0, _pscale), hh[_pmi], ch - hh[_pmi]))
                                _perturb[_pmi, 0] = _pcx; _perturb[_pmi, 1] = _pcy
                            # Use micro_legalize (1nm gap) for perturbation — full legalize
                            # is too slow for dense benchmarks (ibm06 takes minutes).
                            _perturb_t = self._micro_legalize(torch.tensor(_perturb, dtype=torch.float32), benchmark)
                            _perturb = _perturb_t.numpy()
                            T_hsa_current_start = hpwl_T_start * 0.6
                            t_hsa_start = time.time()
                            _rebuild_state(_perturb, _perturb_ori)
                        _reheat_no_improve = 0; _reheat_count += 1
                        print(f"  [hSA] reheat #{_reheat_count} (stagnated {_reheat_K} oracle calls)")
                    # Refresh congestion map every 30 oracle calls
                    if _cong_weights is not None and hsa_oracle % 30 == 0:
                        try:
                            _h2 = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                            _v2 = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                            _t2 = _h2 + _v2
                            _mc2 = np.array([_t2[c] if 0 <= c < len(_t2) else 0.0
                                             for c in [plc.get_grid_cell_of_node(p) for p in _plc_ids]],
                                            dtype=np.float32)
                            _thr2 = float(np.percentile(_mc2, 50))
                            _wts2 = np.maximum(0.0, _mc2 - _thr2).astype(np.float64)
                            if _wts2.sum() > 0:
                                _cong_weights = _wts2 / _wts2.sum()
                        except Exception:
                            pass
                    # Periodic soft macro centroid update (every 100 oracle calls).
                    # Fast O(total_pins) pass to reposition soft macros toward connected hard macros.
                    if n_mac > n_hard and hsa_oracle % 100 == 0:
                        _pos_soft = self._update_soft_macros(best_pos.copy(), benchmark)
                        _cost_soft = compute_proxy_cost(
                            torch.tensor(_pos_soft, dtype=torch.float32), benchmark, plc
                        )["proxy_cost"]
                        oracle_calls += 1; hsa_oracle += 1
                        if _cost_soft < best_cost:
                            best_cost = _cost_soft; best_pos = _pos_soft
                            _rebuild_state(best_pos, best_ori)
                            _reheat_no_improve = 0
                    accepted_since_ckpt = 0

            # Final checkpoint for any pending accepted moves
            if accepted_since_ckpt > 0:
                cost = compute_proxy_cost(
                    torch.tensor(pos, dtype=torch.float32), benchmark, plc
                )["proxy_cost"]
                oracle_calls += 1; hsa_oracle += 1
                if cost < best_cost:
                    best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                    hsa_improved += 1
                    print(f"  [hSA] final {cost:.4f}")

            if hsa_oracle > 0:
                print(f"  [hSA] {hsa_improved} improved, {hsa_accepted}/{hsa_step} accepted, "
                      f"{hsa_oracle} oracle, swap={'on' if _hsa_swap_enabled else 'off'}")

        # ── Oracle SA tail: direct proxy minimization with remaining time ─────
        # hSA uses HPWL surrogate which doesn't correlate for congestion-dominated
        # benchmarks. After hSA stagnates, use oracle (compute_proxy_cost) directly
        # for single-macro moves. Only helps when micro_legalize will succeed (no
        # legalize_hard needed): oracle SA can find positions with harder legalization
        # that look better by proxy but are actually worse after legalize_hard (+0.08 extra).
        _osa_remain = deadline - time.time() - oracle_time_est * 4
        _osa_pre_micro = self._micro_legalize(torch.tensor(best_pos, dtype=torch.float32), benchmark)
        _osa_micro_ok = compute_overlap_metrics(_osa_pre_micro, benchmark)["overlap_count"] == 0
        if _osa_remain > oracle_time_est * 5 and _osa_micro_ok:
            _pre_osa_best_pos = best_pos.copy(); _pre_osa_best_cost = best_cost  # save for revert
            _rebuild_state(best_pos, best_ori)
            # Adaptive scale: large early (broad exploration) → shrink over time (exploitation).
            # Congestion-dominated benchmarks use a larger base scale (macros need bigger moves to
            # escape congested bins; 4% probe would miss bin boundaries).
            _osa_scale_base = max(cw, ch) * (0.08 if _wl_frac < 0.04 else 0.04)
            _osa_T = best_cost * 0.010         # 1% temperature: accepts ≤1% worse moves for exploration
            _osa_improved = 0; _osa_accepted = 0; _osa_tries = 0
            _osa_t0 = time.time(); _osa_deadline = deadline - oracle_time_est * 3
            print(f"  [oSA] {_osa_remain:.0f}s remaining, starting oracle SA tail")
            # Macro selection weights: congestion-biased for congestion-dominated benchmarks.
            # _cong_weights selects macros in high-congestion cells with higher probability,
            # focusing oracle calls where they can reduce the dominant cost component.
            _osa_mv_arr = np.array(movable_hard, dtype=np.int64)
            _osa_probs = (None if _cong_weights is None
                          else _cong_weights.astype(np.float64) / _cong_weights.sum())
            while time.time() < _osa_deadline:
                # Adaptive scale: decay from base to 25% of base as time runs out
                _t_frac = min(1.0, (time.time() - _osa_t0) / max(1e-9, _osa_deadline - _osa_t0))
                _osa_scale = _osa_scale_base * max(0.25, 1.0 - 0.75 * _t_frac)
                if _osa_probs is not None:
                    ci = int(np.random.choice(_osa_mv_arr, p=_osa_probs))
                else:
                    ci = random.choice(movable_hard)
                cx = float(np.clip(pos[ci, 0] + random.gauss(0, _osa_scale), hw[ci], cw - hw[ci]))
                cy = float(np.clip(pos[ci, 1] + random.gauss(0, _osa_scale), hh[ci], ch - hh[ci]))
                _osa_tries += 1
                if _overlaps(ci, cx, cy):
                    continue
                old_x, old_y = float(pos[ci, 0]), float(pos[ci, 1])
                pos[ci, 0] = cx; pos[ci, 1] = cy
                all_cx[ci] = cx; all_cy[ci] = cy
                cost = compute_proxy_cost(
                    torch.tensor(pos, dtype=torch.float32), benchmark, plc
                )["proxy_cost"]
                oracle_calls += 1
                delta = cost - best_cost
                if delta < 0 or (delta < _osa_T and random.random() < math.exp(-delta / _osa_T)):
                    _osa_accepted += 1
                    if cost < best_cost:
                        best_cost = cost; best_pos = pos.copy(); best_ori = ori.copy()
                        _osa_improved += 1
                        _rebuild_fix_bbox_for_nets(macro_to_nets[ci])
                        print(f"  [oSA] {cost:.4f}")
                        # Refresh congestion weights on improvement to track updated hotspots
                        if _osa_probs is not None and _cong_weights is not None:
                            try:
                                _h2 = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
                                _v2 = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
                                _mc2 = np.array([(_h2+_v2)[c] if 0<=c<len(_h2) else 0.0
                                                 for c in [plc.get_grid_cell_of_node(p) for p in _plc_ids]],
                                                dtype=np.float32)
                                _thr2 = float(np.percentile(_mc2, 50))
                                _wts2 = np.maximum(0.0, _mc2 - _thr2).astype(np.float64)
                                if _wts2.sum() > 0:
                                    _cong_weights = _wts2 / _wts2.sum()
                                    _osa_probs = _cong_weights / _cong_weights.sum()
                            except Exception:
                                pass
                else:
                    pos[ci, 0] = old_x; pos[ci, 1] = old_y
                    all_cx[ci] = old_x; all_cy[ci] = old_y
            if _osa_improved > 0 or _osa_accepted > 5:
                print(f"  [oSA] {_osa_improved} improved, {_osa_accepted}/{_osa_tries} accepted")
            # Verify oracle SA improvements survive legalization (prevents ibm06-type regression
            # where oracle SA finds "better" positions with harder legalization cascade)
            if _osa_improved > 0:
                _post_micro = self._micro_legalize(torch.tensor(best_pos, dtype=torch.float32), benchmark)
                if compute_overlap_metrics(_post_micro, benchmark)["overlap_count"] > 0:
                    best_pos = _pre_osa_best_pos; best_cost = _pre_osa_best_cost
                    print(f"  [oSA] reverted: legalization fails for oSA result")
                else:
                    best_pos = _post_micro.numpy()  # use micro-legalized result directly

        elif not _osa_micro_ok and _osa_remain > oracle_time_est * 20:
            # ILS restarts: micro_legalize fails for current best (needs legalize_hard).
            # Try perturbed restarts to find a configuration where micro_legalize succeeds.
            # Key for ibm06: SA tends to push macros to canvas boundaries (creating stuck
            # pairs). Different random perturbations may avoid this.
            _ils_budget_s = max(oracle_time_est * 12, min(oracle_time_est * 30, _osa_remain / 15))
            _ils_scale = max(cw, ch) * 0.10
            _ils_count = 0; _ils_ok_count = 0
            print(f"  [ILS] {_osa_remain:.0f}s, ~{_osa_remain/_ils_budget_s:.0f} restarts planned")
            # Identify boundary-adjacent macros: cascade failures in ibm06 are caused by
            # macros at canvas edges creating stuck fixed-macro overlap pairs.
            _bdy_thresh = max(cw, ch) * 0.05
            _bdy_macros = [i for i in movable_hard
                           if (pos[i,0]-hw[i] < _bdy_thresh or cw-pos[i,0]-hw[i] < _bdy_thresh
                               or pos[i,1]-hh[i] < _bdy_thresh or ch-pos[i,1]-hh[i] < _bdy_thresh)]
            while time.time() < deadline - oracle_time_est * 8:
                _ils_count += 1
                _prt = best_pos.copy()
                _n_p = max(1, n_mv // 12)
                # Alternate: half the restarts target boundary macros (push inward),
                # half use random perturbation to diversify exploration.
                if _bdy_macros and _ils_count % 2 == 0:
                    _targets = random.sample(_bdy_macros, min(max(1, len(_bdy_macros)//2), _n_p))
                    for _pm in _targets:
                        _dx = cw/2 - _prt[_pm, 0]; _dy = ch/2 - _prt[_pm, 1]
                        _d = math.sqrt(_dx*_dx + _dy*_dy)
                        if _d > 0:
                            _prt[_pm, 0] = float(np.clip(_prt[_pm,0]+_dx/_d*_ils_scale, hw[_pm], cw-hw[_pm]))
                            _prt[_pm, 1] = float(np.clip(_prt[_pm,1]+_dy/_d*_ils_scale, hh[_pm], ch-hh[_pm]))
                else:
                    for _pm in random.sample(movable_hard, min(_n_p, n_mv)):
                        _prt[_pm, 0] = float(np.clip(_prt[_pm, 0] + random.gauss(0, _ils_scale), hw[_pm], cw - hw[_pm]))
                        _prt[_pm, 1] = float(np.clip(_prt[_pm, 1] + random.gauss(0, _ils_scale), hh[_pm], ch - hh[_pm]))
                _rebuild_state(_prt, best_ori)
                _r_dead = min(deadline - oracle_time_est * 5, time.time() + _ils_budget_s)
                _r_best = float('inf'); _r_pos = _prt.copy()
                while time.time() < _r_dead:
                    ci = random.choice(movable_hard)
                    cx = float(np.clip(pos[ci, 0] + random.gauss(0, _ils_scale * 0.5), hw[ci], cw - hw[ci]))
                    cy = float(np.clip(pos[ci, 1] + random.gauss(0, _ils_scale * 0.5), hh[ci], ch - hh[ci]))
                    if _overlaps(ci, cx, cy): continue
                    ox, oy = pos[ci, 0], pos[ci, 1]
                    pos[ci, 0] = cx; pos[ci, 1] = cy; all_cx[ci] = cx; all_cy[ci] = cy
                    r_cost = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)["proxy_cost"]
                    oracle_calls += 1
                    if r_cost < _r_best: _r_best = r_cost; _r_pos = pos.copy()
                    else: pos[ci, 0] = ox; pos[ci, 1] = oy; all_cx[ci] = ox; all_cy[ci] = oy
                _r_micro = self._micro_legalize(torch.tensor(_r_pos, dtype=torch.float32), benchmark)
                if compute_overlap_metrics(_r_micro, benchmark)["overlap_count"] == 0:
                    _ils_ok_count += 1
                    _r_mc = compute_proxy_cost(_r_micro, benchmark, plc)["proxy_cost"]
                    oracle_calls += 1
                    if _r_mc < best_cost:
                        best_cost = _r_mc; best_pos = _r_micro.numpy()
                        print(f"  [ILS] restart {_ils_count}: micro OK, {_r_mc:.4f} (new best!)")
                        _osa_micro_ok = True  # final legalization will use micro result
                        break
                _rebuild_state(best_pos, best_ori)
            if _ils_count > 0:
                print(f"  [ILS] {_ils_count} restarts ({_ils_ok_count} micro OK), best={best_cost:.4f}")

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
        # Full legalization if significant (>4nm) overlaps remain (e.g. after DREAMPlace).
        if self._count_significant_overlaps(result, benchmark, threshold=0.004) > 0:
            return self._legalize_hard(result, benchmark)
        # Micro-legalization: fix any residual sub-4nm overlaps.
        # With pre-SA micro_legalize, CT overlaps are resolved before SA sees them;
        # SA's _overlaps filter prevents new overlaps from SA moves. So this should be a no-op.
        micro = self._micro_legalize(result, benchmark)
        # Use exact float64 check (matches harness compute_overlap_metrics).
        if compute_overlap_metrics(micro, benchmark)["overlap_count"] > 0:
            # micro didn't fully converge; start legalize_hard from micro (closer to optimum).
            lh = self._legalize_hard(micro, benchmark)
            return lh
        return micro


