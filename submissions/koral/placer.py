"""
LKPlacer — Five-phase macro placer with electrostatic GP front-end.

Phase α  Focused Electrostatic Global Placement  (gp.run_global_placement)
Phase 0  Legalize hard macros
Phase 1  Build FastEvaluator (bit-exact mirror of PlacementCost)
Phase 2  Lin-Kernighan k-opt + grid sweep
Phase 3  Parallel LAHC polish (true cost via fast evaluator, 16-way parallel search)
"""

from __future__ import annotations

import math
import random
import time
import importlib.util
import concurrent.futures
import multiprocessing
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from macro_place.benchmark import Benchmark

me = Path(__file__).resolve().parent
if str(me) not in sys.path:
    sys.path.append(str(me))

from engine import StaticDesignData, FastEvaluator, lahc_polish

# ────────────────────────────────────────────────────────────────────────────
# Phase 0 — Legalization
# ────────────────────────────────────────────────────────────────────────────


def _overlap_pair(p1, s1, p2, s2):
    dx, dy = abs(p1[0] - p2[0]), abs(p1[1] - p2[1])
    ox, oy = (s1[0] + s2[0]) / 2 - dx, (s1[1] + s2[1]) / 2 - dy
    return (max(0, ox), max(0, oy)) if ox > 0 and oy > 0 else (0.0, 0.0)


def _has_overlap(i, pos, sizes):
    n = pos.shape[0]
    for j in range(n):
        if i == j: continue
        ox, oy = _overlap_pair(pos[i], sizes[i], pos[j], sizes[j])
        if ox > 0 and oy > 0: return True
    return False


def _spiral_search(i, pos, sizes, movable, canvas_w, canvas_h, gap=1e-3):
    cx, cy = pos[i]; w, h = sizes[i]; hw, hh = w/2, h/2
    step = min(w, h) * 0.1
    for r in range(1, 100):
        for dx, dy in [(r, 0), (-r, 0), (0, r), (0, -r), (r, r), (r, -r), (-r, r), (-r, -r)]:
            nx, ny = max(hw + gap, min(canvas_w - hw - gap, cx + dx * step)), max(hh + gap, min(canvas_h - hh - gap, cy + dy * step))
            pos[i] = [nx, ny]
            if not _has_overlap(i, pos, sizes): return [nx, ny]
    return [cx, cy]


def _legalize(pos, sizes, movable, canvas_w, canvas_h, gap=1e-3):
    n = pos.shape[0]
    for _ in range(50):
        any_ov = False
        for i in range(n):
            if not movable[i]: continue
            for j in range(n):
                if i == j: continue
                ox, oy = _overlap_pair(pos[i], sizes[i], pos[j], sizes[j])
                if ox > 0 and oy > 0:
                    any_ov = True; dx, dy = pos[i] - pos[j], np.sign(pos[i] - pos[j])
                    if ox < oy: pos[i, 0] += dy[0] * (ox + gap)
                    else: pos[i, 1] += dy[1] * (oy + gap)
                    pos[i, 0] = max(sizes[i, 0]/2 + gap, min(canvas_w - sizes[i, 0]/2 - gap, pos[i, 0]))
                    pos[i, 1] = max(sizes[i, 1]/2 + gap, min(canvas_h - sizes[i, 1]/2 - gap, pos[i, 1]))
        if not any_ov: break
    for i in range(n):
        if movable[i] and _has_overlap(i, pos, sizes): pos[i] = _spiral_search(i, pos, sizes, movable, canvas_w, canvas_h, gap)
    return pos

# ────────────────────────────────────────────────────────────────────────────
# Parallel Refinement Logic
# ────────────────────────────────────────────────────────────────────────────

def _run_single_lahc_worker(benchmark: Benchmark, data: StaticDesignData, init_pos: np.ndarray, list_len: int, time_budget_s: float, seed: int):
    ev = FastEvaluator(benchmark, data); ev.restore(init_pos)
    out = lahc_polish(ev, list_len=list_len, time_budget_s=time_budget_s, seed=seed)
    return {"proxy_cost": out["proxy_cost"], "positions": ev.positions.copy(), "iters": out["iters"]}

def parallel_lahc_polish(benchmark: Benchmark, data: StaticDesignData, init_pos: np.ndarray, list_len=100, time_budget_s=600.0, n_chains=16, base_seed=0, verbose=True):
    if verbose: print(f"  [PAR-LAHC] launching {n_chains} chains in parallel for {time_budget_s:.0f}s", flush=True)
    ctx = multiprocessing.get_context('spawn')
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_chains, mp_context=ctx) as executor:
        futures = [executor.submit(_run_single_lahc_worker, benchmark, data, init_pos, list_len, time_budget_s, base_seed + i*100) for i in range(n_chains)]
        best_cost, best_pos, total_iters = float("inf"), init_pos.copy(), 0
        for f in concurrent.futures.as_completed(futures):
            try:
                res = f.result(); total_iters += res["iters"]
                if res["proxy_cost"] < best_cost: best_cost, best_pos = res["proxy_cost"], res["positions"]
            except Exception as e:
                if verbose: print(f"    [PAR-LAHC] Worker failed: {e}", flush=True)
    if verbose: print(f"  [PAR-LAHC] all done. total iters={total_iters}  best={best_cost:.4f}", flush=True)
    return {"proxy_cost": best_cost, "positions": best_pos, "iters": total_iters}

# ────────────────────────────────────────────────────────────────────────────
# LKPlacer Orchestrator
# ────────────────────────────────────────────────────────────────────────────

def _load_plc(name: str):
    from macro_place.loader import load_benchmark, load_benchmark_from_dir
    root = Path("external/MacroPlacement/Testcases/ICCAD04") / name
    if root.exists(): return load_benchmark_from_dir(str(root))[1]
    ng45 = {"ariane133": "ariane133", "ariane136": "ariane136", "nvdla": "nvdla", "mempool_tile": "mempool_tile"}
    d = ng45.get(name.replace("_ng45", ""))
    if d:
        base = Path("external/MacroPlacement/Flows/NanGate45") / d / "netlist" / "output_CT_Grouping"
        if (base / "netlist.pb.txt").exists(): return load_benchmark(str(base / "netlist.pb.txt"), str(base / "initial.plc"))[1]
    return None

class LKPlacer:
    def __init__(self, seed=42, time_budget_s=3000.0, run_gp=True, gp_pop_size=4, gp_steps=500, lahc_list_len=100, verbose=True):
        self.seed, self.time_budget_s, self.run_gp, self.gp_pop_size, self.gp_steps, self.lahc_list_len, self.verbose = seed, time_budget_s, run_gp, gp_pop_size, gp_steps, lahc_list_len, verbose
        self.gp_budget_s = 90.0

    def _log(self, msg):
        if self.verbose: print(f"[lk_placer] {msg}", flush=True)

    def place(self, benchmark: Benchmark) -> torch.Tensor:
        from macro_place.objective import compute_proxy_cost
        import sys; me = Path(__file__).resolve().parent
        if str(me) not in sys.path: sys.path.append(str(me))
        t0 = time.time(); random.seed(self.seed); np.random.seed(self.seed); torch.manual_seed(self.seed); plc = _load_plc(benchmark.name)

        if self.run_gp:
            try:
                spec = importlib.util.spec_from_file_location("lk_placer_gp", str(me / "gp.py"))
                gp_mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(gp_mod)
                self._log(f"Phase α: focused electrostatic global placement (pop={self.gp_pop_size}, steps={self.gp_steps}, budget={self.gp_budget_s:.0f}s)")
                gp_positions = gp_mod.run_global_placement(benchmark, plc, pop_size=self.gp_pop_size, n_steps=self.gp_steps, time_budget_s=self.gp_budget_s, seed=self.seed, verbose=self.verbose)
                benchmark.macro_positions = torch.from_numpy(gp_positions).float()
            except Exception as e: self._log(f"Phase α: SKIPPED due to exception: {e}")

        self._log("Bootstrap: extracting design data and discarding C++ oracle")
        data = StaticDesignData.extract(benchmark, plc)
        n_hard, cw, ch = benchmark.num_hard_macros, float(benchmark.canvas_width), float(benchmark.canvas_height)
        sizes_np, mov_np, init = benchmark.macro_sizes[:n_hard].cpu().numpy().astype(np.float64), benchmark.get_movable_mask()[:n_hard].cpu().numpy().astype(bool), benchmark.macro_positions[:n_hard].cpu().numpy().astype(np.float64)

        self._log("Phase 0: legalizing hard macros")
        hard_legal = _legalize(init, sizes_np, mov_np, cw, ch)
        pos_full = benchmark.macro_positions.numpy().astype(np.float64).copy(); pos_full[:n_hard] = hard_legal
        benchmark.macro_positions = torch.from_numpy(pos_full).float()

        self._log("Phase 1: building FastEvaluator")
        ev = FastEvaluator(benchmark, data); c0 = ev.proxy_cost()
        self._log(f"  fast baseline: proxy={c0['proxy_cost']:.4f} wl={c0['wirelength_cost']:.4f} den={c0['density_cost']:.4f} cong={c0['congestion_cost']:.4f}")
        true_c = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
        self._log(f"  oracle: {true_c['proxy_cost']:.4f}  overlaps={true_c['overlap_count']}")
        best_true, best_pos = (float(true_c["proxy_cost"]), ev.positions.copy()) if true_c["overlap_count"] == 0 else (float("inf"), None)

        remaining = max(60.0, self.time_budget_s - (time.time() - t0))
        self._log(f"Phase 3: Parallel LAHC polish, budget={remaining:.0f}s")
        out = parallel_lahc_polish(benchmark, data, ev.positions, list_len=self.lahc_list_len, time_budget_s=remaining, n_chains=16, base_seed=self.seed, verbose=self.verbose)
        self._log(f"  LAHC: best={out['proxy_cost']:.4f}  total_iters={out['iters']}"); ev.restore(out["positions"])
        
        final_tc = compute_proxy_cost(torch.from_numpy(ev.positions).float(), benchmark, plc)
        if final_tc["overlap_count"] == 0 and final_tc["proxy_cost"] < best_true: best_true, best_pos = float(final_tc["proxy_cost"]), ev.positions.copy()
        self._log(f"DONE  best_true={best_true:.4f}  time={time.time()-t0:.1f}s")
        return torch.from_numpy(best_pos if best_pos is not None else ev.positions).float()
