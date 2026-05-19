# Design: Custom Differentiable Global Placer

## Problem

Xplace ePlace finds a WL+Density minimum and ignores congestion. We need a GP that
directly minimizes the proxy cost scoring function:

```
Proxy = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```

This is the exact function we're scored on. Nothing we have currently optimizes it.

## Core Idea

Run gradient descent directly on macro positions with the proxy cost as the loss.
The proxy cost has three terms — all three can be made differentiable in PyTorch.
Adam optimizer. Multiple random restarts for diversity. No external tools. No Docker
rebuild.

## The Three Terms

### 1. Wirelength (HPWL)

HPWL is not differentiable (max/min operations). Standard fix: log-sum-exp smooth
approximation, used by every academic placer since ePlace (2014).

```
HPWL_smooth(net) = α × log(Σ exp(x_i/α)) - α × log(Σ exp(-x_i/α))   [+ same for y]
```

α is a temperature parameter that controls smoothness. PyTorch has `torch.logsumexp`.
This is the exact same approximation DREAMPlace and Xplace use internally — it's
well-understood and works well.

**What we use:** `torch.logsumexp` on pin positions, one call per net per axis.
Pin positions = macro center + pin offset from `benchmark.macro_pin_offsets`.
For soft macros (no offsets): use center directly.

### 2. Density Penalty

This is the spreading term — prevents all macros from piling into one spot.

ePlace's original approach (GPU Poisson solver via FFT) is complex. For macro
placement with 100–1000 large blocks (not millions of standard cells), a simpler
bin-based approach is sufficient:

For each macro, compute its overlap with each bin in a `grid_rows × grid_cols` grid.
Sum overlapping area per bin. Penalty = sum over bins where density exceeds target.

**What we use:** Differentiable bin density via smooth overlap computation. Each macro
contributes to nearby bins based on a cosine or triangle kernel (common in academic
placers as a simpler alternative to ePlace's Bell-shaped function). Fully implemented
in PyTorch, ~30 lines.

Alternatively: reuse `plc.get_density_cost()` for validation, but compute gradients
via PyTorch autograd on our own density implementation.

### 3. Congestion (RUDY)

RUDY (Rectangular Uniform wire Density) — the same metric the TILOS evaluator uses.

For each net with bounding box W×H:
- Each gcell overlapping the bbox receives routing demand proportional to 1/(W×H)
- Compare demand to capacity: `h_cap = hroutes_per_micron × gcell_h`
- Congestion cost = sum of demand/capacity excess (or use the TILOS formula directly)

**What we use:** Compute net bboxes via `torch.min/max` over pin positions (same pin
positions as WL term). Map to gcell grid. Compute demand. All differentiable.
The capacity values come directly from `benchmark.hroutes_per_micron` and
`benchmark.vroutes_per_micron` which are already loaded.

## Legality

During gradient descent, macros will overlap. Two options:

**Option A (preferred):** Rely on the density penalty to prevent overlap. When
density weight is high enough, the penalty pushes macros apart naturally — this is
exactly how ePlace works. No explicit legalization during optimization.

**Option B (fallback):** Run `_legalize_hard` every N steps (e.g., every 100
iterations) and continue optimizing from legalized positions. Slower but guaranteed
legal at each checkpoint.

After optimization ends, always run `_legalize_hard` + check overlaps before passing
to SA.

## Optimizer

Adam with learning rate warmup. Standard in all modern differentiable placers.
Position variable = `[num_macros, 2]` tensor with `requires_grad=True`.
Clamp to canvas bounds after each step (simple box constraint).
Fixed macros: zero out their gradients before optimizer step.

## Multiple Seeds / Restarts

Run the optimizer from multiple starting positions:
- CT positions (1 seed) — the reference
- CT positions + Gaussian noise at various scales — same noise schedule as Xplace
- Pure random positions within canvas — for exploring new basins

Pick best result by oracle proxy cost after legalization.

This directly addresses the local minima problem: diverse starts + congestion in the
objective = some seeds will find genuinely low-congestion placements.

## Integration into Pipeline

```
CT positions
    ↓
Custom GP (multiple seeds, ~300–600s total)
    - Optimizes 1.0×WL + 0.5×Density + 0.5×Congestion
    - Adam, ~2000–5000 iterations per seed
    - Pick best seed by oracle cost
    ↓
_legalize_hard
    ↓
Sequential SA (remaining budget, ~2800–3000s)
```

Can also feed Xplace GP positions as one of the seeds — Xplace gives good WL/Density
starting points, and custom GP then adds congestion optimization on top.

## What We Already Have

- `benchmark.net_nodes`, `benchmark.macro_pin_offsets` — pin-level connectivity
- `benchmark.hroutes_per_micron`, `benchmark.vroutes_per_micron` — routing capacity
- `benchmark.grid_rows`, `benchmark.grid_cols` — gcell grid dimensions
- `_legalize_hard` — overlap resolution after GP
- `compute_proxy_cost` — for evaluation/seed selection
- `plc.get_horizontal_routing_congestion()` — for validation against our RUDY
- FastEvaluator — for quick proxy estimates during seed selection

## Expected Results

| Stage | ibm01 proxy | Avg (17 benchmarks) |
|-------|-------------|---------------------|
| Xplace GP (current) | 0.908 | ~1.25 |
| Custom GP (estimated) | 0.82–0.88 | ~1.00–1.10 |
| + SA | 0.76–0.82 | ~0.93–1.02 |

Rank 1 is 0.9671 average. Getting to 0.93–1.02 is rank 3–10 territory.

Uncertainty is real: depends on how well the density approximation spreads macros and
how much congestion the optimizer can reduce without sacrificing WL.

## Risks

| Risk | Mitigation |
|------|-----------|
| Density term too weak → macros overlap | Increase density weight; tune target_density |
| Density term too strong → ignores WL/congestion | Weight schedule: anneal density weight down over iterations |
| Slow per seed | Tune iterations; use fewer seeds if needed |
| RUDY gradient too small vs WL | Scale congestion weight up (0.5 → 1.0 or higher) |
| Result worse than Xplace | Fall back to Xplace result (keep both, pick better) |

## What NOT To Do

- Do NOT try to port ePlace's full FFT Poisson solver — overkill for macro placement
- Do NOT remove Xplace from the pipeline — use it as a fallback and as one of the seeds
- Do NOT use the TILOS `compute_proxy_cost` for gradient computation — it's not
  differentiable (Python + C++ evaluator). Only use it for evaluation.
