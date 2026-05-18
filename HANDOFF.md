# HANDOFF: Macro Placement Challenge 2026

**Deadline: May 21, 2026 | Prize: $49K | Submission: https://forms.gle/YDRtYV5Vq68SZgKW9**

## Current Status (2026-05-18)

- **Estimated avg score: ~1.42–1.46** (rank ~30–40)
- **Target top 5: ~1.04** — gap is ~0.4 points
- **Critical insight: The gap comes from analytical placement.** Fix DREAMPlace or integrate Xplace.

---

## Leaderboard Context

| Rank | Team | Score | Method |
|------|------|-------|--------|
| 1 | Carrotato | 0.9671 | Triton + **Xplace** |
| 5 | Cezar | ~1.037 | Unknown |
| 10 | KLA MACH | 1.1764 | **DREAMPlace** + CD + SA + ILS |
| ~35 | Us | ~1.42–1.46 | CT + parallel SA + ibm06 fix |

The top teams use analytical placers as global init for **all** benchmarks. We can't because:
1. DREAMPlace (new May 2026 version) is broken for our use case (see below)
2. Xplace not yet integrated

---

## Verified Benchmark Scores

| Benchmark | Score | Method | Notes |
|-----------|-------|--------|-------|
| ibm01 | 0.9197 | Old DREAMPlace + SA | Best ever; new DREAMPlace gives 1.01–1.23 |
| ibm01 (current) | ~1.03 | CT + swap-SA | DREAMPlace stochastic, often loses to CT 1.04 |
| ibm02 | 1.5476 | CT + SA | Baseline; DREAMPlace center-init gives 1.5575 (sometimes) |
| ibm06 | 1.6877 | CT + perturbed SA | CASCADE ESCAPE: 1.83→1.69 via 3-4.5% perturbation |
| ibm09 | 1.1126 | CT + SA | Near-optimal |
| Average (all 17) | ~1.457 | Old code baseline | |

**Full run in progress:** container `5ee5e2a7c07f` (started May 18, ~17 hours total).
New code has: swap moves in oracle SA, center-init DREAMPlace, perturbed SA workers.

---

## The DREAMPlace Problem (Session 2 Findings)

### Why DREAMPlace is broken in the current Docker image

The Dockerfile clones **HEAD** of DREAMPlace which is a **May 2026 version** with new features:
- **Entropy injection** — fires when overflow > 95% at start of center-init (ALL macros at center → always fires → scrambles macros)
- **Aggressive divergence detection** — rolls back to suboptimal positions
- **Quadratic penalty** — triggers `quad_penalty_coeff` AttributeError (fixed but wrong method)

The old Docker image (built earlier May 17-18) used an OLDER DREAMPlace that didn't have these features and gave ibm01=0.9197.

### What we tried and why it didn't work

| Config | ibm01 result | ibm02 result | Problem |
|--------|-------------|-------------|---------|
| center-init, 2000+3000 GPU iters | 2.79 | 1.94 | 140–200 hard macro overlaps → proxy blows up |
| center-init, 500+1000 GPU iters | 1.01–1.23 (stochastic) | 1.55–1.60 | Unpredictable; 75% of runs worse than CT |
| CT-init, 500+1000 | 1.60 | — | 139 overlaps → always discarded |
| CT-init, 2000+3000 | 1.95 | — | 200 overlaps |
| legalize_flag=1 | 1.31 | Hannan FAIL | Snaps to grid → destroys WL optimization |
| macro_place_flag=1 | — | Hannan FAIL | Same Hannan grid problem |
| multi-seed (3 seeds) | All 3 worse than CT | — | Wasted 3 min; P(any beats CT) only ~25% |

**Root cause of overlap problem:** DREAMPlace's bin-based density penalty doesn't prevent individual macro overlaps. When optimization runs too long (>500 iterations), macros get pushed together by WL gradients and the density bins aren't fine enough to prevent overlap at macro boundaries.

### Partial fix in current code

- `stop_overflow=0.001` → disables divergence rollback
- `500+1000 iterations` → reduces overlap creation
- `center_init=True` for ALL benchmarks → CT-init always creates overlaps
- `PlaceObj.obj_fn` patched to fix `quad_penalty_coeff` AttributeError

Result: DREAMPlace center-init now sometimes (25% of runs) beats CT. Rest of the time, SA falls back to CT automatically.

---

## Highest Priority Next Steps

### 1. ⚡ Pin DREAMPlace to Working Commit (HIGH IMPACT, 45 min)

In `submissions/koral/Dockerfile`, pin to a commit before entropy_injection was added:

```dockerfile
RUN git clone https://github.com/limbo018/DREAMPlace.git /tmp/dreamplace_src \
    && cd /tmp/dreamplace_src \
    && git checkout <COMMIT_HASH>  # commit from ~May 2024 or earlier
```

How to find the commit: look for the commit that ADDED `entropy_injection` in NonLinearPlace.py.
Everything BEFORE that commit should give the old behavior (ibm01=0.9197).

Lines to look for in NonLinearPlace.py (if present → version too new):
```python
entropy_injection(self.pos[0], placedb, ...)
```

**Expected impact:** ibm01 goes from ~1.03 to ~0.92 (matching old result).
All ibm02-18 get proper analytical init. Average might drop to ~1.35.

### 2. ⚡ Integrate Xplace (HIGH IMPACT, 2-4 hours)

Rank-1 team (Carrotato) uses Xplace from https://github.com/cuhk-eda/Xplace
Add to Dockerfile:
```dockerfile
RUN git clone --depth=1 https://github.com/cuhk-eda/Xplace.git /tmp/xplace \
    && cd /tmp/xplace && pip install -r requirements.txt && python setup.py install
```

Call as a Python module (Xplace is PyTorch-based), use output as SA init.
Xplace is likely much better than DREAMPlace for this specific problem.

### 3. 🔧 Better Legalization (MEDIUM IMPACT, 2-3 hours)

Current `_legalize_hard` uses greedy push with O(n²) Python loops — stalls at 100-200 overlaps.

Better approaches:
- **Random kick + restart**: move one stuck macro far away, re-run
- **Scipy minimize** with overlap penalty
- **Vectorized** numpy legalization (all pairs simultaneously, not Gauss-Seidel)

### 4. 🔧 GPU-Accelerated Proxy for SA (MAJOR, 1+ days)

The bottleneck: each oracle SA call takes ~0.5s (TILOS C++ evaluator).
If we implement HPWL + density + congestion in PyTorch tensors, SA runs on GPU → 50x faster.
This would give ibm01 ~0.85-0.90 range.

### 5. 🔧 RePlAce via OpenROAD subprocess

Call OpenROAD's RePlAce as a subprocess. Large dependency (~2GB) but well-documented.

---

## Current Pipeline

```
CT positions (baseline ~1.04–1.57 per benchmark)
    ↓
DREAMPlace center-init (stochastic; if beats CT, use it)
    ↓
Best of (CT, DREAMPlace) → 4 parallel SA workers
    ↓  (perturbed: 0, 1.5%, 3%, 4.5% sigma)
CD → FD → hSA → oracle SA with swap moves
    ↓
Final score
```

**ibm06 special case**: perturbed workers (3-4.5%) escape the CT cascade where
47 sub-4nm overlaps chain-react during micro_legalize. Non-perturbed workers stay stuck.

---

## Technical Reference

### Docker dev loop
```bash
# Build (takes ~40 min to compile DREAMPlace)
docker build -t koral-placer -f submissions/koral/Dockerfile .

# Run single benchmark with bind-mount (no rebuild needed for code changes)
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none koral-placer submissions/koral/placer.py -b ibm01

# Run all 17
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/path/to/repo:/challenge" --network none \
  koral-placer submissions/koral/placer.py --all
```

### DREAMPlace parameters (in _run_dreamplace)
- `gp_noise_ratio=0.025` (center-init) / `0.01` (CT-init)
- `stop_overflow=0.001` (center-init, disables rollback) / `0.03` (CT-init)
- Bins: 128×128 (stage 1) → 512×512 (stage 2)
- `macro_halo_x/y=50` (5nm gap prevents float-precision boundary overlaps)
- 500 iters stage 1 + 1000 iters stage 2 (GPU)

### Key files
- `placer.py`: KoralPlacer class, all algorithms
- `bookshelf.py`: Benchmark → DREAMPlace adapter
- `Dockerfile`: build environment
- `patch_dreamplace.sh`: CUDA 12.4 patches (keep pin_pos_cuda, disable pin_pos_cuda_segment)

### Judge hardware
AMD EPYC 9655P, 16 cores, 100GB RAM, NVIDIA RTX 6000 Ada 48GB

### Scoring
`proxy_cost = 1.0×WL + 0.5×Density + 0.5×Congestion` (minimize)
Any hard macro overlap → benchmark disqualified.
