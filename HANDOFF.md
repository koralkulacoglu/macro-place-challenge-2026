# Macro Placement Challenge 2026 — Strategic Handoff

**Deadline: May 21, 2026** (≈4 days). Prize: $49K. Submission form: https://forms.gle/YDRtYV5Vq68SZgKW9

---

## Competition Basics

**Objective:** Minimize avg proxy cost across 17 IBM benchmarks (ibm01–ibm18, skipping ibm05).

```
Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```

Hard constraint: **zero hard macro overlaps** in all 17 benchmarks, or you're DQ'd entirely.

**Tier 2** (top 7 only): Judges run OpenROAD PnR on 4 NG45 designs. Weighted geometric mean of WNS:3, TNS:2, Area:1.

---

## Leaderboard Context (as of May 17)

| Rank | Team | Avg Proxy | Notes |
|------|------|-----------|-------|
| 1 | Carrotato | 0.9671 | Triton kernels + Xplace + polish |
| 2 | Shoom | 0.978 | MultiDREAMPlace (DP + CD refinement) |
| 3 | vmallela | 1.0109 | Verified |
| 4 | DREAMPlaceProMaxUltra | 1.0121 | Verified, 6h total |
| 5 | Cezar | 1.037 | Team-provided Dockerfile |
| 6 | thinkorplace | 1.0771 | Cascading Saddle Escape (CD+LNS+SA+Hessian) |
| 10 | KLA MACH | 1.1764 | DREAMPlace + parallel CD/SA + ILS |
| — | RePlAce baseline | 1.4578 | Verified |
| — | **us (current est.)** | **~1.4–1.5** | Unverified; ibm01=0.9221 is good |

**Pattern in top 10:** Every single one uses analytical placement (DREAMPlace or similar) as starting point for ALL benchmarks. Our main gap is that DREAMPlace currently only helps ibm01.

---

## Current Algorithm: `submissions/koral/placer.py`

### Pipeline

```
Benchmark
  │
  ├─ CT positions (from .plc file) → _legalize_hard
  │
  ├─ [if DREAMPlace available] center-init DREAMPlace → _legalize_hard
  │    Compare with CT, keep best.
  │    DREAMPlace only helps ibm01 (all ibm02-18 diverge 4-8×).
  │
  └─ _cd_lns_polish (600s budget):
       1. CD (coordinate descent, HPWL-guided, oracle check per pass, exit after 5 no-improve)
       2. LNS (pairwise swaps in spatial clusters, adaptive cap: 15-50% of budget)
       3. FD gradient descent (finite-difference proxy gradient for top-20 congested macros)
       4. Oracle SA (remaining time, k=1 single-macro Gaussian moves, T-annealed)
       5. Soft macro update (try _update_soft_macros, revert if no gain)
       6. Skip final legalization if zero f32-detectable overlaps
```

### How to Run

```bash
# Local (no DREAMPlace — falls back to CT+SA, fine for ibm02-18)
uv run evaluate submissions/koral/placer.py -b ibm02

# All 17 benchmarks
uv run evaluate submissions/koral/placer.py --all

# Docker (required for ibm01 DREAMPlace)
docker build -t koral-placer -f submissions/koral/Dockerfile .
docker run --rm --runtime=nvidia --gpus all \
  -v $(pwd):/challenge --network none \
  --entrypoint python koral-placer \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01
```

---

## Best Known Scores

These are verified results from before the last session's unfinished runs:

| Benchmark | Our Best | CT Baseline | SA Baseline | RePlAce | Notes |
|-----------|----------|-------------|-------------|---------|-------|
| ibm01 | **0.9221** | 2.0463 | 1.3166 | 0.9976 | DREAMPlace; better than RePlAce |
| ibm02 | 1.5476 | 2.0431 | 1.9072 | 1.8370 | k=1 oracle SA |
| ibm03 | 1.3244 | 2.1484 | 1.7401 | 1.3222 | ~RePlAce level |
| ibm04 | 1.3104 | 1.9321 | 1.5037 | 1.3024 | ~RePlAce level |
| ibm06 | 1.9249 | 2.1620 | 2.5057 | 1.6187 | **Problem child** — see below |
| ibm07 | 1.4650 | 2.1144 | 2.0229 | 1.4633 | ~RePlAce level |
| ibm09 | 1.1126 | 1.6728 | 1.3875 | 1.1194 | ~RePlAce level |
| ibm10-18 | unknown | 2.0–2.8 | varies | 1.5–1.8 | Never run with new code |

**Estimated average (ibm01–09 known + ibm10-18 rough guess):** ~1.45–1.55.

### ibm06 Problem
CT positions evaluate at 1.6577 (52 sub-4nm overlaps that TILOS ignores but our legalizer does not). After `_legalize_hard`, it becomes 1.9249 — a +0.27 proxy cost hit that we never recover. The 32 overlapping macro pairs from legalization persist through the entire optimization. Oracle SA with 460s budget finds small improvements (~1.90–1.92 range).

---

## Key Code Details

### Files
- `submissions/koral/placer.py` — main placer (~1291 lines)
- `submissions/koral/bookshelf.py` — Benchmark → DREAMPlace Bookshelf format adapter
- `submissions/koral/Dockerfile` — clones DREAMPlace at build time, applies CUDA 12.4 patches
- `submissions/koral/patch_dreamplace.sh` — CUDA 12.4 + NumPy 2.0 patches

### Key Parameters (`_cd_lns_polish`)
- `k_osa_max = 1`: Single-macro oracle SA moves (cluster sizes cause >95% rejection on ibm02-range)
- `lns_frac = max(0.15, min(0.50, _wl_frac * 3.0))`: Adaptive LNS time cap
- `_swap_enabled`: Auto-enabled when Gaussian valid rate < 3% (dense benchmarks)
- Oracle: `compute_proxy_cost(pos_tensor, benchmark, plc)` — ~0.5s/call

### Oracle Call Rate
~0.5s per oracle call → ~1200 calls in 600s budget. This is the fundamental bottleneck. Competitors at rank 1–10 either:
a) Use faster surrogate (HPWL-only), or
b) Get much better starting positions from analytical placement, then refine with fewer oracle calls

---

## What's Been Tried / Empirical Findings

### DREAMPlace
- **ibm01**: center-init = 0.9221, CT = 1.04 → DREAMPlace wins by 10%
- **ibm02-18**: center-init = 4–8× worse than CT → DREAMPlace diverges for dense benchmarks
- Root cause: DREAMPlace optimizes HPWL + density, not TILOS proxy. For dense benchmarks starting from scratch, it ends up with poor congestion, high density. CT positions already encode congestion awareness from Circuit Training's RL training.
- **CT-init DREAMPlace (untested)**: Set `random_center_init_flag=0` and pass CT positions as initial placement. This could improve ibm02-18 by refining CT from a better starting point.

### SA Oracle
- k=1 single-macro moves: ~1200 effective oracle calls/600s for ibm02 (271 movable macros)
- k=6 cluster moves: ~40 effective oracle calls/600s (95% rejection from overlaps) — **DO NOT USE**
- Swap moves: always valid, but less effective than Gaussian for sparse benchmarks
- ibm09 (dense, 0 free space): oracle SA finds only 5 improvements in 580s → CT already near-optimal

### LNS
- HPWL-guided swaps in k=8 spatial clusters
- Consistently finds 0.001–0.003 improvements for CT-dominated benchmarks
- ibm06: LNS was consuming 570/600s budget (many HPWL improvements) → adaptive cap added (15%)

### FD Gradient
- Works for WL-dominated benchmarks: ibm02 (+2 improvements), ibm04 (step 0 = 1.3094 vs 1.3104)
- Ineffective for pure congestion benchmarks: ibm06 (gnorm≈0 for congestion-only gradient)

### What Doesn't Work
- k_osa_max > 1 for ibm-range benchmarks (all suffer from low valid rate)
- DREAMPlace center-init for ibm02-18
- Running `plc.optimize_stdcells()` in the loop (too slow, ~2min for ibm06)
- Directed swap as primary SA move for sparse benchmarks

---

## Strategic Recommendations (Priority Order)

### 1. CT-init DREAMPlace for All Benchmarks (Highest Impact)

Change `random_center_init_flag=0` and initialize DREAMPlace from CT positions:

```python
# In _run_dreamplace, add before writing bookshelf:
# Write CT positions to .pl file instead of random center
params_dict["random_center_init_flag"] = 0
# Write the actual CT coordinates to the .pl file in write_bookshelf()
```

In `bookshelf.py → write_bookshelf()`: write CT center coordinates into the `.pl` file so DREAMPlace starts from CT rather than the origin. This is the single highest-leverage change because:
- CT positions are congestion-aware (trained by RL)
- DREAMPlace can refine HPWL+density starting from that good point
- Top-2 teams are doing exactly this

Estimated impact: ibm02–18 could drop from ~1.3–1.9 to ~1.1–1.5 range if it works.

### 2. HPWL-Oracle Hybrid SA (Multiplies Throughput by ~500×)

Replace most oracle calls with fast HPWL evaluation. Use proxy oracle only every N=50 accepted moves:

```python
# Fast inner SA (HPWL oracle, ~0.001s/call → 600,000 calls/600s)
for inner_step in range(50):
    pick macro, generate Gaussian candidate
    if delta_hpwl < 0 or SA accept: apply move
# Every 50 accepted: call proxy oracle to verify and update best
cost = compute_proxy_cost(...)
if cost < best: update best
else: revert to last_oracle_pos
```

This changes the SA from "1200 proxy oracle calls" to "1200 proxy verifications + 60,000 HPWL-guided moves", dramatically improving search coverage. The WL-dominated benchmarks (ibm02, ibm03, ibm04, ibm07) would benefit most.

### 3. Multi-Start with Best-of-N

Run the full CD+LNS+oracle pipeline N=2–3 times with different random seeds, keep best result. Budget: 600s / N per run. Costs nothing in terms of new code; estimates improvement via randomness diversity.

### 4. ibm06-Specific Fix

ibm06's fundamental problem: CT positions have 52 sub-4nm overlaps that TILOS ignores but our legalizer resolves by pushing macros, costing +0.27 proxy. Options:
- Try `GAP=0` in `_legalize_hard` (4nm separation instead of 5nm) — might let CT positions through with fewer moves
- Skip legalization entirely for benchmarks where CT evaluator reports 0 overlaps and we detect only sub-threshold ones
- Run oracle SA immediately on unlegal CT positions, using only TILOS-valid acceptance criterion

### 5. Tune Existing Code on ibm10–18

The new code (k_osa_max=1, adaptive LNS, FD, skip-legalize) hasn't been run on ibm10-18. Run all benchmarks to get a score baseline before attempting new approaches.

---

## Immediate Next Steps (In Order)

1. **Run baseline**: `uv run evaluate submissions/koral/placer.py --all` (17h, ~600s/bench) to get current avg score
2. **CT-init DREAMPlace**: Modify `write_bookshelf()` to write CT positions to `.pl` file, set `random_center_init_flag=0`, test ibm02 via Docker. If ibm02 DP beats CT, this is the path.
3. **HPWL-hybrid SA**: Implement as described above, test on ibm02/ibm06
4. **ibm06 legalize fix**: Try `GAP=2` (just above TILOS 4nm threshold) to minimize CT position distortion
5. **Docker build**: The Dockerfile is functional. Build with `docker build -t koral-placer -f submissions/koral/Dockerfile .` (~40 min, requires internet for DREAMPlace clone)

---

## CT-init DREAMPlace: How to Implement

In `submissions/koral/bookshelf.py`, the `write_bookshelf()` function writes macro positions. Find where it writes the `.pl` file (position file) and ensure it uses `benchmark.macro_positions` (CT positions) rather than origin:

```python
# bookshelf.py: write_bookshelf() already writes CT positions to .pl file
# The key change is in placer.py _run_dreamplace():
params_dict["random_center_init_flag"] = 0   # use .pl positions, not center
# This tells DREAMPlace to start from whatever .pl says (CT positions)
```

Currently `center_init=True` in `_run_dreamplace` always sets `random_center_init_flag=1`. Change the call to use `center_init=False` and verify the `.pl` file has CT positions.

Check `bookshelf.py` → `write_bookshelf()` to confirm CT positions are written. They likely are already since fixed macros need correct positions.

---

## Docker Notes

```bash
# Build (needs internet; clones DREAMPlace from GitHub)
docker build -t koral-placer -f submissions/koral/Dockerfile .

# Run with live repo mount + GPU
docker run --rm --runtime=nvidia --gpus all \
  -v $(pwd):/challenge --network none \
  --entrypoint python koral-placer \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01

# Full evaluation (17 benchmarks, ~17h)
docker run --rm --runtime=nvidia --gpus all \
  -v $(pwd):/challenge --network none \
  --entrypoint python koral-placer \
  -m macro_place.evaluate submissions/koral/placer.py --all
```

CUDA 12.4 patches applied at build time (in `patch_dreamplace.sh`): cmake detection fix, libcuda.so stub symlink, 4 CUDA targets disabled (CUB API incompatible), `np.string_` → `np.bytes_`.

---

## Benchmark Characteristics

| Benchmark | Hard Macros | Movable | Notes |
|-----------|-------------|---------|-------|
| ibm01 | ~12 | small | Sparse — DREAMPlace beats CT |
| ibm02 | ~276 | 271 | Medium — oracle SA helps |
| ibm03 | ~230 | ~220 | WL-dominated |
| ibm04 | ~300 | ~295 | WL-dominated |
| ibm06 | ~1200 | ~52 | **Congestion-dominated**; CT has sub-4nm overlaps |
| ibm07 | ~220 | ~200 | ~RePlAce level |
| ibm09 | ~270 | ~260 | Very dense; CT near-optimal |
| ibm10 | ~786 | high | Slow (~3600s legalization) |
| ibm12-18 | unknown | unknown | Never run with current code |

---

## Architecture Constraints

- Judging hardware: AMD EPYC 9655P, 16 cores, 100GB RAM, RTX 6000 Ada 48GB
- Runtime: `--network none` (no internet at runtime)
- Time limit: 1 hour/benchmark, 17 hours total
- Docker image must include all deps (DREAMPlace, etc.)
- Planner image base: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime`
