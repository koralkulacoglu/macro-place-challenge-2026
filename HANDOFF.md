# HANDOFF: Macro Placement Challenge 2026

**Deadline: May 21, 2026 | Prize: $49K | Submission: https://forms.gle/YDRtYV5Vq68SZgKW9**

---

## Current Status (2026-05-20, afternoon)

**Pipeline**: CT positions → Xplace multi-seed GP + RUDY v3 → parallel SA (16 workers)

**Git head**: `f800731` (main, local only — not pushed to remote yet pending confirmation)

**Full `--all` run in progress**: `bench_final_v2` container, started ~2026-05-20 midday.

| Benchmark | Result | Notes |
|-----------|--------|-------|
| ibm01 | **0.8850** ✅ | 274 seeds, 1800s Xplace, 1800s SA (16 workers) |
| ibm02 | **1.5307** ⚠️ | Damaged — Xplace lost to double-patching bug, SA from CT 1.5658 |
| ibm03 | in progress | Best GP oracle 1.1521 (CT=1.3255, 13% improvement) |
| ibm04–17 | pending | ~1hr each |

**Expected avg**: ibm01 excellent (0.8850 < rank 1 avg 0.9671), ibm02 badly hurt (+0.27 vs expected), ibm03–17 with RUDY v3. Estimated final avg somewhere in 1.05–1.15 range.

---

## Leaderboard (May 18, 2026)

| Rank | Team | Avg Score | Method |
|------|------|-----------|--------|
| 1 | Carrotato | **0.9671** | Triton + Xplace + congestion-aware GP |
| 2 | Shoom | 0.978 | MultiDREAMPlace + CD |
| 3 | vmallela | 1.0109 | Fast evaluator + good GP |
| 5 | Cezar | 1.037 | Custom refinement |
| 10 | KLA MACH | 1.1764 | DREAMPlace+CD+SA+ILS |
| ~15–25 | **Us (estimated)** | 1.05–1.15 | Xplace + RUDY v3 + SA |

---

## Current Pipeline (full detail)

```
CT positions (benchmark.macro_positions from initial.plc)
    ↓
Xplace multi-seed GP (up to 1800s or 400 seeds)
    - Adaptive noise schedule: [0.05, 0.10, 0.13, 0.15, 0.18, 0.20]
      (lower: [0.02, 0.03, 0.05, 0.07, 0.10, 0.13] for hyperconnected nets_per_macro > 60)
    - stop_overflow=0.05, inner_iter=8000, mixed_size=True
    - RUDY v3 congestion gradient (corner scatter, weight=0.1, fires at overflow<0.08, K=50 cache)
    - Best seed by oracle cost after legalization; top-16 kept for SA
    ↓
_legalize_hard (greedy micro-legalization)
    ↓
Parallel SA (16 workers, ~1800s budget)
    - Workers pre-forked in __init__ before CUDA (fixes CUDA+fork deadlock)
    - CD → LNS → FD → hSA (Boltzmann in fast-eval space) → oSA
    - FastEvaluator: 383x speedup
```

**Total budget per benchmark**: 3600s (1800s Xplace + 1800s SA for fast benchmarks).

---

## RUDY Congestion Gradient (implemented and working)

Injected into Xplace GP via source patches. Three versions developed:

### v1 (failed)
- Loss = sum(demand_map²) where demand scatters HPWL to net centers with autograd
- Problem: gradient of HPWL = WL gradient → clusters macros → oracle cost increases

### v2 (correct direction, deployed for ibm01)
- Step 1: compute demand map (DETACHED — no autograd)
- Step 2: interpolate congestion at each node's position WITH autograd
- Gradient: ∂loss/∂pos = local congestion slope → pushes nodes toward lower congestion
- ibm01 result: 274 seeds, best oracle 0.8855, SA final 0.8850

### v3 (current — corner scatter, live ibm03+)
- Same two-step structure as v2
- Step 1 change: scatter HPWL/4 to each of 4 bbox CORNERS instead of net center
- Why better: corners are where routing demand actually lives (pin endpoints)
- Two nets with overlapping bboxes get congestion at shared corners → more accurate hot spots
- ibm03 best seed: 1.1521 (CT=1.3255, 13.4% improvement)

### Xplace patch mechanism
`submissions/koral/xplace_patches/apply_patches.py` patches 3 files at runtime:
1. `src/calculator.py` — inject RUDY gradient call after WL grad
2. `main.py` — add `--rudy_weight`, `--rudy_start_iter` args
3. `src/param_scheduler.py` — add `use_rudy`, `rudy_weight` fields

`apply_patches.py` is idempotent (checks if `new` content already present before patching).
`KoralPlacer._xplace_patched` class variable prevents re-patching within a process.
`rudy_loss.py` is copied from bind mount to `/opt/xplace/src/core/` at KoralPlacer init.

---

## Key Bugs Found and Fixed

### 1. CUDA+fork deadlock
- **Symptom**: ibm02+ hang indefinitely after ibm01 completes
- **Root cause**: fork() with hot CUDA context causes futex deadlock in child
- **Fix**: Pre-fork 16 SA workers in `KoralPlacer.__init__()` before any CUDA usage
- **Evidence**: `[KoralPlacer] 16 SA workers pre-forked (CUDA-clean at init)` in logs

### 2. RUDY double-patching (corrupts main.py)
- **Symptom**: ibm02 Xplace fails with `argparse.ArgumentError: conflicting option string: --rudy_weight`
- **Root cause**: KoralPlacer re-initializes for each benchmark (SA workers run full KoralPlacer per benchmark), applying patches 16× and adding duplicate `--rudy_weight` args
- **Fix 1**: `apply_patches.py` checks `if new in txt: skip`
- **Fix 2**: `KoralPlacer._xplace_patched = True` class guard (per-process)
- **Damage**: ibm02 lost its Xplace phase; SA from CT 1.5658 → 1.5307

### 3. RUDY v1 gradient direction
- **Symptom**: oracle cost increases with RUDY enabled (1.469 vs 1.403 baseline for seed=42)
- **Root cause**: autograd through HPWL computation = WL gradient (clusters macros)
- **Fix**: detach demand map; gradient flows only through node position interpolation

---

## Hard Constraints (do not change)

- `stop_overflow=0.01` → crashes Xplace LP legalization (PulpError NaN/inf). Use 0.05.
- `use_route_force=True` → crashes GPU pattern router (1.8M segments overflow CUDA buffer). Do NOT attempt.
- CUDA+fork: never fork after CUDA is initialized. Pre-fork in __init__ only.
- `apply_patches.py` must remain idempotent (double-application must be safe).

---

## Verified Xplace GP Scores (old baseline, before RUDY — 6 seeds, ~36s)

| Benchmark | CT baseline | Old GP best | New GP best (RUDY v2/v3) |
|-----------|------------|-------------|--------------------------|
| ibm01 | 1.039 | 0.908 | **0.8855** (274 seeds, v2) |
| ibm02 | 1.566 | 1.552 | 1.5307 (SA only, bug) |
| ibm03 | 1.326 | 1.176 | **1.1521** (v3, in progress) |
| ibm04 | 1.313 | 1.115 | pending |
| ibm06 | 1.658 | 1.410 | pending |
| ibm07 | 1.476 | 1.270 | pending |
| ibm08 | 1.466 | 1.339 | pending |
| ibm09 | 1.113 | 0.952 | pending |
| ibm10 | 1.340 | 1.135 | pending |
| ibm11 | 1.214 | 0.979 | pending |
| ibm12 | 1.625 | 1.369 | pending |
| ibm13 | 1.385 | 1.068 | pending |
| ibm14 | 1.594 | 1.392 | pending |
| ibm15 | 1.603 | 1.517 | pending |
| ibm16 | 1.491 | 1.325 | pending |
| ibm17 | 1.739 | 1.469 | pending |
| ibm18 | 1.790 | 1.764 | pending |

---

## What We Tried That Didn't Work

### Xplace route_force
Fundamental incompatibility with IBM ICCAD04. Two approaches tried:
1. Bookshelf mode: `gpugr` rejects non-lefdef format unconditionally
2. Synthetic LEF/DEF (183 layers): database loads, GP works, but route_force SIGSEGV during GPU pattern routing — IBM benchmarks generate 1.8M segments overflowing a hard-coded CUDA kernel buffer

**Do NOT attempt again.**

### RUDY v1
Autograd through HPWL = WL gradient. Wrong direction.

---

## Next Steps (if time allows before deadline)

1. **Wait for bench_final_v2 to complete** (~14 more hours from now)
2. **Submit results** from the run
3. **Potential second run improvements**:
   - Adaptive `rudy_weight` per benchmark based on congestion fraction from CT baseline
   - Cap Xplace to fewer seeds for ultra-fast benchmarks → give SA more time
   - True RUDY (scatter to all bbox cells, not just corners) — most like rank 1

---

## Running the Current Submission

```bash
# The current run: bench_final_v2
docker logs bench_final_v2 --tail=50

# To start a fresh full run (after bench_final_v2 completes)
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none \
  -e KORAL_RUDY_WEIGHT=0.1 \
  --entrypoint bash koral-placer-xplace \
  -c "cd /challenge && python3 -m macro_place.evaluate submissions/koral/placer.py --all 2>&1 | tee /tmp/bench_all.log"
```

---

## Files

| File | Purpose |
|------|---------|
| `submissions/koral/placer.py` | `KoralPlacer` main (~1800 lines) |
| `submissions/koral/bookshelf.py` | Benchmark → ISPD2005 bookshelf for Xplace |
| `submissions/koral/fast_eval.py` | FastEvaluator (383x SA speedup) |
| `submissions/koral/Dockerfile` | Docker: Xplace + dependencies |
| `submissions/koral/xplace_patches/rudy_loss.py` | RUDY v3 congestion loss (corner scatter) |
| `submissions/koral/xplace_patches/apply_patches.py` | Idempotent Xplace source patcher |

---

## Submission

When ready: https://forms.gle/YDRtYV5Vq68SZgKW9
