# LKPlacer — Team IDK Submission

**Team IDK** — Ian Patrick Tan ([@IanPTan](https://github.com/IanPTan)),
Dev ([@Dev077](https://github.com/Dev077)),
Koral Kulacoglu ([@koralkulacoglu](https://github.com/koralkulacoglu))

Five-phase macro placer combining a focused electrostatic global placer
with Lin-Kernighan k-opt swaps and Late Acceptance Hill Climbing (LAHC).
Pure PyTorch + NumPy, no external tools, no Docker build required.

## Files

| File | Purpose |
|---|---|
| `placer.py` | `LKPlacer` class + `FastEvaluator` + LK + LAHC bodies |
| `gp.py` | Phase α focused electrostatic global placement |
| `README.md` | This file |

The submission is just these three files. No other directories, no
binaries, no patches. `placer.py` dynamically loads `gp.py` from the
same directory via `importlib`.

## Running

The harness auto-discovers the first class in `placer.py` that exposes
a `place(self, benchmark) -> torch.Tensor` method (`LKPlacer`).

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
  VRAM (~2 GB). Falls back to CPU gracefully if no GPU is available
  (Phase α GP becomes slower but still completes).
- Network: `--network none` is fine; no network calls are made.

## Time budget

Default `time_budget_s = 3300.0` (55 min), leaving 5 min margin below
the competition's 1 h-per-benchmark cap. Phase budgets within that:

| Phase | Budget | Notes |
|---|---|---|
| α — Electrostatic GP | 90 s | 4 replicas with replica-exchange |
| 0 — Legalize | <5 s | Greedy push-apart + spiral fallback |
| 1 — FastEvaluator build + calibrate | ~5 s | One-time |
| α₂ — True-cost subgradient | 60 s | Stochastic discrete subgradient |
| 2 — Lin-Kernighan k-opt | ~30–120 s | 3 passes, ends when convergent |
| 3 — LAHC polish | remaining (~50 min) | The bulk of refinement |

## Algorithm

### Phase α — Focused Electrostatic Global Placement (`gp.py`)

Continuous global placer running in float32 on GPU. Three innovations
vs. classical ePlace / DREAMPlace:

1. **Focused Poisson density target.** Standard ePlace pushes every cell
   toward uniform density. Our proxy cost (the actual judging metric)
   only penalises the top 10 % of density cells. We set the Poisson
   source = `density − top10pct_threshold`, clamped at zero. Only hot
   cells generate a field, concentrating force where it lowers true
   cost.
2. **Focused congestion gradient.** Same idea applied to RUDY routing
   demand: only cells above the top-5 % threshold contribute to the loss.
3. **Pure 2-D FFT Poisson solver** for the density potential. Provides
   long-range global forces from any hot spot to anywhere else on the
   canvas — far more responsive than a local bilinear-gradient method.

Runs `pop_size = 4` replicas in parallel; periodically swaps poorly
performing replicas with stronger ones (replica exchange, "REX"). After
the 90 s budget, all replicas are legalized and the best by oracle proxy
cost is selected.

### Phase 0 — Legalization

Greedy push-apart with spiral fallback. Pairwise overlap resolution at
the macro level, then a deterministic spiral search around the canvas
for any macro that couldn't be resolved locally.

### Phase 1 — FastEvaluator

Bit-exact mirror of `macro_place.objective.compute_proxy_cost`, ~250×
faster on a full re-eval and ~8000× faster on per-move WL deltas.
Cached per-net pin tables, prefix-sum congestion grids, and incremental
HPWL re-computation. Calibrated against the oracle with a 5-sample
linear fit at the start; aborts to anchor placement if Pearson r < 0.85.

### Phase α₂ — Stochastic True-Cost Subgradient

Discrete propose-and-test moves on the calibrated FastEvaluator. Each
proposal is a single-macro Gaussian translation with overlap rejection;
accepted iff the FastEvaluator delta is negative. Cheap enough to run
~5–10 k iterations in 60 s.

### Phase 2 — Lin-Kernighan k-opt Swaps

Macro priority queue (highest fast-cost contribution first). For each
macro, attempt chain-depth k-opt swaps with its 24 nearest neighbors,
accepting any chain that reduces the fast proxy. Three passes; typically
converges in 2–3 passes.

### Phase 3 — LAHC (Late Acceptance Hill Climbing)

The deep-refinement workhorse — runs for the remainder of the budget
(~50 min, ~1–1.5 M iterations). Mixed move set:

- Hard-macro Gaussian translation with overlap rejection
- Hard-macro swap (preserves density distribution)
- Soft-macro Gaussian translation
- Partner-centroid biased proposal (move a macro toward the centroid
  of its net neighbors — accelerates HPWL reduction)

Acceptance criterion (LAHC): accept if new cost ≤ cost from
`L = 100` iterations ago. This produces effective annealing without
explicit temperature tuning. Best-ever placement is tracked separately
and returned at the end.

## Why this design

The proxy cost is dominated by density and congestion (each weighted
0.5×), with wirelength only contributing ~5–10 % on most IBM benchmarks.
A pure HPWL-minimizing placer leaves the largest cost components on the
table. The focused-Poisson GP attacks density and congestion directly;
LAHC's iterative refinement (over a million moves at ~30 μs each via
FastEvaluator) extracts the last few percent of every component.

The architectural choice that matters most is **putting the compute
where the cost lives**: ~3 % of the budget goes to parallel GP
exploration, ~97 % goes to single-trajectory LAHC polishing. Earlier
attempts at this submission spent most of the budget on parallel GP
candidates that all converged to the same basin — wasted compute.

## Verified scores

Run inside the standard pytorch image, single benchmark each:

| Benchmark | Proxy | Time |
|---|---|---|
| ibm12 | **1.2035** | 50.3 min |
| ariane133 (NG45) | **0.6424** | 55.0 min |

## Constraints respected

- **Zero hard-macro overlaps** at output (final legalize before return)
- **All macros within canvas bounds**
- **Fixed macros unmoved** (`benchmark.macro_fixed` honored)
- **Output is center coordinates in microns**, `[num_macros, 2]` float32
- **No design-specific hardcoding** — adaptations key off structural
  properties (canvas size, macro count) where used, not benchmark names

## Contact

Koral Kulacoglu — kulacoglukoral@gmail.com (primary contact)
Ian Patrick Tan — [@IanPTan](https://github.com/IanPTan)
Dev — [@Dev077](https://github.com/Dev077)
