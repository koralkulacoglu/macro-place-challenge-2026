# HANDOFF: Macro Placement Challenge 2026

**Deadline: May 21, 2026 | Prize: $49K | Submission: https://forms.gle/YDRtYV5Vq68SZgKW9**

---

## Current Status (2026-05-20, evening)

**Pipeline**: `GraphGradPlacer` — custom GPU-batched analytical placer, pure PyTorch.
No Xplace, no SA, no custom Docker deps.

**Git head**: `030c7e5` (main, pushed)

**Verified scores:**

| Benchmark | Score | Notes |
|-----------|-------|-------|
| ibm01 | **0.8663** ✅ | joint-release, 165s |
| ibm03 | **1.0687** ✅ | joint-release, 165s |
| Full `--all` baseline | **1.2240 avg** | pre joint-release, 73 min |
| Full `--all` joint-release | **pending** | expect ~1.19–1.21 |

---

## Leaderboard (May 18, 2026)

| Rank | Team | Avg | Method |
|------|------|-----|--------|
| 1 | Carrotato | **0.9671** | Triton + Xplace + congestion-aware GP |
| 2 | Shoom | 0.978 | MultiDREAMPlace + CD |
| 3 | vmallela | 1.0109 | Fast evaluator + GP |
| 5 | Cezar | 1.037 | Custom refinement |
| 10 | KLA MACH | 1.1764 | DREAMPlace+CD+SA+ILS |
| ~12–18 | **Us** | **1.22 → ~1.20** | GraphGradPlacer (see below) |

---

## Pipeline (full detail)

```
benchmark.macro_positions (initial.plc)
    ↓
_legalize(hard) — greedy push-apart + spiral fallback
    ↓
Phase A: soft-only Adam, steps [0, 4000)
    - pop_size=96 candidates, all locked at hard_legal_t
    - soft init: initial.plc soft positions + randn * 0.1 µm
    - loss = (wl_norm + 0.5·density + 0.5·RUDY_cong).sum()
    - pop.grad[:, :n_hard].zero_() each step; hard reasserted
    ↓
Phase B: joint-release, steps [4000, 5000)
    - Hard UNlocked
    - loss += alpha_anchor * normalized_anchor_pull (alpha=50)
    - loss += alpha_ov * normalized_overlap_area (alpha=1000)
    - _legalize every 200 steps to drain accumulated overlap
    - hard clamped to canvas each step
    ↓
Final _legalize for all 96 candidates
    ↓
Surrogate ranking: topk(-surr, 48)
    ↓
True-cost eval: compute_proxy_cost on top-48
    ↓
Anchor safety net: compare against legalized initial.plc
    → return best
```

**Surrogate** (mirrors `compute_proxy_cost` weights exactly):
- `wl_norm`: WAHPWL per net, summed, / ((cw+ch)*n_nets). Gamma anneals 1.0→0.05 (sharp Manhattan).
- `density`: TILOS top-10% grid-cell mean × 0.5.
- `RUDY_cong`: differentiable RUDY with TILOS H/V track normalization × 0.5.

**Why two-phase?** Phase A lets soft macros converge quickly to the net-topology basin without hard movement interfering. Phase B then lets hard make small corrective moves — the anchor pull keeps them near initial.plc (which is a strong prior), the overlap penalty prevents overlaps, and periodic _legalize cleans up residual overlap. Empirically: density drops 2-3%, congestion drops 1%, proxy improves 0.8-2.2%.

---

## Key Design Decisions

### Why not Xplace+SA (previous approach)?
Xplace+SA (KoralPlacer) is gone. Issues:
- Required custom Dockerfile (+40 min build), RUDY patches, bookshelf conversion
- CUDA+fork deadlock (SA workers + CUDA)
- Double-patching bug corrupted ibm02
- Final scores with all that: ibm01 0.8850, avg est. ~1.15
- GraphGradPlacer: ibm01 0.8663, avg 1.2240, 165s/bench, zero deps

### Why not full joint mode (`_place_joint`)?
`_place_joint` (dead code, kept for reference) uses exponential `alpha_ov` ramp → 5000. At that scale the overlap penalty completely dominates proxy loss. Hard macros park in non-overlapping positions that aren't optimal for WL/density/cong. The two-phase approach is strictly better: lock→converge→release with gentle penalties.

### Why does locking hard help Phase A?
Soft macros start at initial.plc soft positions (good prior) and converge to their optimal positions given the fixed hard layout. This is a convex-ish subproblem. Joint mode makes it non-convex from step 0 — harder to optimize.

### Why does releasing hard help Phase B?
Cost breakdown on ibm01: wl=0.071 (8%), den=0.503 (29%), cong=1.088 (63%). Density and congestion depend strongly on hard layout. Even small hard adjustments reduce hot spots.

---

## Key Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| `pop_size` | 96 | fills RTX 6000 Ada VRAM comfortably |
| `soft_steps` | 5000 | ~165s/bench; budget is 3600s |
| `soft_lr` | 0.01 | Adam learning rate |
| `release_step` | 4000 (80%) | joint phase = last 1000 steps |
| `alpha_anchor` | 50.0 | normalized anchor pull on hard |
| `alpha_ov` | 1000.0 | normalized overlap penalty |
| `relegalize_every` | 200 | steps between _legalize calls in Phase B |

---

## Empirical Findings

| Finding | Detail |
|---------|--------|
| Hard lock is necessary for Phase A | Without it, joint mode from step 0 loses ~15% vs lock-then-release |
| Joint release improves density/cong | ibm01: den 0.517→0.503 (-3%), cong 1.097→1.088 (-1%); ibm03: -2.2% total |
| 5000 steps uses ~4.5% of budget | ~165s/bench of 3600s limit — significant headroom for more steps |
| Surrogate is faithful | Ranking by `wl_norm + 0.5·dens + 0.5·cong` correlates well with true proxy |
| Anchor safety net triggers rarely | Anchor (legalized initial.plc) is ~1.03–1.04 on IBM benchmarks |

---

## Files

| File | Purpose |
|------|---------|
| `submissions/koral/placer.py` | `GraphGradPlacer` — sole submission file (~1250 lines) |
| `HANDOFF.md` | This file |
| `CLAUDE.md` | Codebase guide for Claude Code |

Dead files (kept in repo history, not used):
- `submissions/koral/bookshelf.py` — Xplace bookshelf format converter
- `submissions/koral/fast_eval.py` — SA FastEvaluator
- `submissions/koral/Dockerfile` — Xplace Docker image
- `submissions/koral/xplace_patches/` — Xplace RUDY patches

---

## Next Steps

**Immediate (before deadline):**
1. Run `--all` with joint-release to get final avg score
2. Submit via https://forms.gle/YDRtYV5Vq68SZgKW9

**If time allows:**
1. **More steps**: `soft_steps=10000` uses ~330s/bench, still well within budget. Phase A gets 8000 steps of convergence, Phase B gets 2000 for hard refinement.
2. **Earlier release**: Try `release_step = 70%` (3500 steps) — gives Phase B 1500 steps. May help congestion-dominated benchmarks more.
3. **Lighter anchor / more hard freedom**: Reduce `alpha_anchor` from 50→20. Lets hard move further from initial.plc; risky if initial.plc is weak, valuable if it's suboptimal.

---

## Running the Submission

```bash
# Single benchmark
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none --entrypoint bash koral-placer-xplace \
  -c "cd /challenge && python3 -m macro_place.evaluate submissions/koral/placer.py -b ibm01"

# Full --all
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none --entrypoint bash koral-placer-xplace \
  -c "cd /challenge && python3 -m macro_place.evaluate submissions/koral/placer.py --all"
```

No `docker build` needed — `koral-placer-xplace` image is reused for its PyTorch version but Xplace is never invoked. `placer.py` is bind-mounted and live.
