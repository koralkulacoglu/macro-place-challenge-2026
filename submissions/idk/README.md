# GraphGradPlacer — Team IDK Submission

**Team IDK** — Ian Patrick Tan ([@IanPTan](https://github.com/IanPTan)),
Dev ([@Dev077](https://github.com/Dev077)),
Koral Kulacoglu ([@koralkulacoglu](https://github.com/koralkulacoglu))

Analytical macro placer combining a TILOS-faithful gradient-based global
placement (soft macros only, Adam optimizer) with Late Acceptance Hill Climbing
(LAHC) deep polish. Pure PyTorch + NumPy, no external tools, no Docker build
required.

## Files

| File | Purpose |
|---|---|
| `placer.py` | `GraphGradPlacer` class + `FastEvaluator` + LAHC body |
| `gp.py` | Older focused electrostatic GP module (not actively used) |
| `README.md` | This file |

The submission is just these files. `placer.py` is the entry point; the
harness auto-discovers `GraphGradPlacer` (the first class with a `place` method).

## Running

**Single benchmark:**
```bash
uv run evaluate submissions/idk/placer.py -b ibm01
```

**All 17 IBM benchmarks (Tier 1):**
```bash
uv run evaluate submissions/idk/placer.py --all
```

**NG45 commercial designs (Tier 2):**
```bash
uv run evaluate submissions/idk/placer.py --ng45
```

**With visualization:**
```bash
uv run evaluate submissions/idk/placer.py --all --vis
# Saves vis/<benchmark>.png with placement + density + congestion panels
```

## Environment

- Runs in `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (the standard
  judges' image) with no Dockerfile or rebuild needed.
- Dependencies: `numpy`, `torch` — already provided by the base image
  and `macro_place` package installation.
- Hardware: targeted for NVIDIA RTX 6000 Ada (48 GB) but uses only modest
  VRAM (~2 GB). Falls back to CPU gracefully if no GPU is available.
- Network: `--network none` is fine; no network calls are made.

## Time budget

Default `time_budget_s = 3300.0` (55 min), leaving 5 min margin below
the competition's 1 h-per-benchmark cap. Phase budgets within that:

| Phase | Budget | Notes |
|---|---|---|
| GP — Soft-only Adam gradient descent | ≤7 min | `gp_max_budget_s=420s`; hard macros locked |
| FastEvaluator build + calibrate | ~5 s | One-time setup |
| LAHC polish | remaining (~48 min) | The bulk of refinement |

## Algorithm

### Phase 1 — Soft-Only Analytical Global Placement

Hard macros are legalized once from `initial.plc` and locked in place.
Only soft macros (standard-cell cluster abstractions) move, optimized
via Adam with a TILOS-faithful differentiable surrogate:

```
loss = wl_normalized + 0.5 * tilos_density + 0.5 * tilos_rudy
```

- **`tilos_wl_normalized`** — WAHPWL via log-sum-exp, normalized exactly
  as `PlacementCost.get_cost()` normalizes HPWL. As γ → 0 this converges
  to true Manhattan HPWL.
- **`tilos_density_loss`** — Exact port of `PlacementCost.get_density_cost`:
  bilinear macro-area spread into a grid; `0.5 × mean(top-10% cells)`.
  Differentiable via min/max + sort.
- **`tilos_rudy_normalized`** — RUDY routing-congestion surrogate with the
  same per-axis H/V track normalization, 1-D smoothing kernel, and top-5%
  mean as the TILOS congestion metric.

Runs `pop_size = 96` parallel soft-macro layouts; after Adam converges
(or the 7 min GP budget is hit), ranks candidates by surrogate and scores
the top-`gp_k_eval = 24` against the true oracle. The best zero-overlap
result (or the legalized anchor if none beats it) is handed to LAHC.

### Phase 2 — FastEvaluator

Bit-exact NumPy reimplementation of `PlacementCost.get_cost /
get_density_cost / get_congestion_cost` with incremental update support.
A single `move_macro()` call costs ~2 ms vs ~4000 ms for the oracle
(~2000× speedup). Built once at the start of LAHC; calibrated with 5
oracle samples; aborts to anchor if Pearson r < 0.85.

### Phase 3 — LAHC Polish

Deep refinement for the remaining ~48 min (~1–1.5 M iterations). Mixed
move set:

- **Hard-macro slide** — Gaussian translation with overlap rejection
- **Hard-macro swap** — nearest-neighbor swap, preserves density
- **Soft-macro translate** — uniform or partner-centroid biased proposal

Acceptance criterion (LAHC): accept if new cost ≤ cost from `L = 100`
iterations ago. No explicit temperature tuning. Best-ever placement is
tracked separately and returned at the end; only committed if zero-overlap
and strictly better than the GP output.

## Why this design

The proxy cost is dominated by density and congestion (each weighted 0.5×);
wirelength contributes ~5–10 % on most IBM benchmarks. Locking hard macros
during GP and letting soft macros absorb the density/congestion gradient
avoids the legalization thrash that plagues joint-optimization approaches.
The bulk of the budget then goes to LAHC, which achieves ~30 μs/iteration
via FastEvaluator and systematically extracts the remaining improvement
across all three cost components.

## Constraints respected

- **Zero hard-macro overlaps** at output (legalized before GP hand-off;
  LAHC only proposes overlap-free moves for hard macros)
- **All macros within canvas bounds**
- **Fixed macros unmoved** (`benchmark.macro_fixed` honored)
- **Output is center coordinates in microns**, `[num_macros, 2]` float32
- **No design-specific hardcoding**

## Contact

Koral Kulacoglu — kulacoglukoral@gmail.com (primary contact)
Ian Patrick Tan — [@IanPTan](https://github.com/IanPTan)
Dev — [@Dev077](https://github.com/Dev077)
