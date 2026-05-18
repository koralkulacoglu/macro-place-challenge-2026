# HANDOFF: Macro Placement Challenge 2026

**Deadline: May 21, 2026 | Prize: $49K | Submission: https://forms.gle/YDRtYV5Vq68SZgKW9**

## Current Status (2026-05-18 Session 3)

- **Estimated avg score: ~1.35–1.45** (rank ~30–40 before today's improvements)
- **Target top 5: ~1.04** — gap is ~0.3–0.4 points
- **Critical gap: global placement quality.** CT starts at 1.04–1.87 per benchmark.
  vmallela (rank 3) achieves 0.76 on ibm01 — primarily through better global macro arrangement.

---

## Leaderboard (updated May 18, 2026)

| Rank | Team | Score | Key differentiator |
|------|------|-------|--------------------|
| 1 | Carrotato | 0.9671 | Triton + **Xplace** |
| 2 | Shoom | 0.978 | MultiDREAMPlace + CD |
| 3 | vmallela | 1.0109 | Fast evaluator + better global placement |
| 4 | DREAMPlaceProMaxUltra | 1.0121 | DREAMPlace + Docker |
| 5 | Cezar | 1.037 | Custom refinement |
| 6 | thinkorplace | 1.0771 | CD + LNS + SA + Hessian |
| ~35 | **Us** | ~1.35–1.42 | CT + fast SA (post session 3) |

**vmallela ibm01 = 0.7644** (vs our 1.039 CT baseline). The 0.28 gap is almost entirely congestion.

---

## Cost Breakdown (IBM Benchmarks, CT baseline)

| Benchmark | proxy | wl% | den | cong | n_hard |
|-----------|-------|-----|-----|------|--------|
| ibm01 | 1.039 | 6.2% | 0.812 | 1.137 | 246 |
| ibm02 | 1.566 | 4.8% | 0.729 | 2.254 | 271 |
| ibm03 | 1.326 | 6.0% | 0.732 | 1.760 | 290 |
| ibm04 | 1.313 | 5.4% | 0.782 | 1.704 | 295 |
| ibm06 | 1.658 | 3.8% | 0.722 | 2.467 | 178 |
| ibm09 | 1.113 | 5.1% | 0.836 | 1.275 | 253 |
| ibm10 | 1.340 | 5.3% | 0.668 | 1.870 | 786 |

**ALL benchmarks are congestion-dominated (WL = 4-6%).** Reducing congestion is the only path to top 5.

---

## Session 3 Accomplishments (May 18, 2026)

### 1. FastEvaluator (submissions/koral/fast_eval.py) — COMPLETE
- 383x faster than oracle on ibm01 (evaluate), 4061x (delta_wl)
- 763x faster on ibm02, 17068x delta_wl
- Calibration r=0.98+ on all tested benchmarks
- Wired into oSA tail: 235K+ fast moves per 3480s budget vs ~144 oracle moves
- **Congestion map API**: `congestion_map()`, `macro_congestion_score()` for gradient moves
- In oSA: 10% of moves now target the most-congested macro specifically

### 2. DREAMPlace Entropy Injection Fix — COMPLETE (in Docker)
- Root cause found: entropy injection fires at overflow > 0.95 (ALWAYS fires for center-init)
- Fix: runtime-patch NonLinearPlace.py at container startup
- **Verified result**: ibm01 DREAMPlace center-init = 1.0214 (vs CT 1.0385) — confirmed working
- Combined with existing Lgamma-divergence patch

### 3. Xplace Integration — IN PROGRESS
- `_run_xplace()` method added to placer.py (subprocess call, bookshelf → .pl parse)
- Dockerfile updated with Xplace build (cmake fix: torch.version.cuda not torch.cuda.is_available())
- **Build approach**: docker exec in running container (avoids WSL2 BuildKit crashes)
- Container `xplace_build` is compiling (99% done as of writing)
- **After build**: `docker commit xplace_build koral-placer-xplace` then test on ibm01

### 4. oSA Improvements
- Temperature annealing: 0.001×cost → 0.01×cost (linear decay)
- Full fast.evaluate for congestion-dominated benchmarks (not delta_wl which is WL-only)
- Guided net-adjacent swap (50% probability of picking net neighbors)
- Congestion-gradient macro selection: 10% of oSA picks highest-congestion macro

---

## Current Pipeline

```
CT positions (baseline ~1.04–1.66 per benchmark)
    ↓ Try Xplace (routability-driven GP, needs Docker GPU)
    ↓ Try DREAMPlace (center-init, patched entropy injection)
    ↓ Best of (Xplace, DREAMPlace, CT) → legalize
    ↓
4 parallel SA workers (perturbed 0, 1.5%, 3%, 4.5% sigma)
    ↓
CD (coordinate descent, delta-HPWL oracle) → LNS → FD → hSA → fast oSA
    ↓
Final micro-legalize
```

---

## Active Benchmark Runs (as of writing)

| Benchmark | Log | Status | Note |
|-----------|-----|--------|------|
| ibm01 | C:/tmp/ibm01_v4.log | oSA running (3127s remaining) | Should finish ~1 hr |
| ibm09 | C:/tmp/ibm09_v4.log | hSA finding improvements (1.1065 so far) | |
| ibm02 | C:/tmp/ibm02_v4.log | FD gradient descent | Started later |
| ibm06 | C:/tmp/ibm06_v4.log | Starting | Cascade escape needed |

---

## Immediate Next Steps

### 1. ⚡ Commit Xplace container as image (5 min after build completes)
```bash
# In container xplace_build, after XPLACE_BUILD_DONE appears:
docker commit xplace_build koral-placer-xplace

# Test Xplace on ibm01:
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none --entrypoint python koral-placer-xplace \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01
```

### 2. ⚡ Run DREAMPlace + SA on ibm01 in Docker (verify 1.02 → <1.00 with full SA)
```bash
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/...:/challenge" --network none --entrypoint python koral-placer \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01
```

### 3. Monitor oSA improvement on ibm01
The new fast oSA is running 3100+ seconds with full_eval mode on ibm01.
Should find improvements that raw hSA misses (hSA uses HPWL surrogate, blind to congestion).

### 4. Submit current best before trying risky changes
Branch `strategy-may18` is the submission branch. Before major refactors:
1. Run a quick 5-benchmark spot check
2. If avg < 1.40: submit immediately

---

## Key Files

- `submissions/koral/placer.py` — KoralPlacer (1400+ lines)
- `submissions/koral/fast_eval.py` — FastEvaluator (calibrated 300-17000x speedup)
- `submissions/koral/bookshelf.py` — Benchmark → Bookshelf adapter (DREAMPlace + Xplace)
- `submissions/koral/Dockerfile` — Docker build with DREAMPlace + Xplace

## Docker

```bash
# Current working image (DREAMPlace only):
docker images | grep koral  # → koral-placer (47dbdb1720e4)

# After Xplace build completes:
docker commit xplace_build koral-placer-xplace

# Run with GPU + live code mount:
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none --entrypoint python koral-placer-xplace \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01
```

## WSL2 Docker Issue

Docker Desktop uses WSL2 which crashes during heavy CUDA compilation (NVCC kills WSL2 VM).
**Workaround that works**: `docker exec -d` in a running container (not `docker build`).
This bypasses BuildKit and the WSL2 crash pattern.

When WSL2 crashes, run:
```
Start-Process "C:\Program Files\Docker\Docker\Docker Desktop.exe"
# Wait ~30s then: until docker ps > /dev/null 2>&1; do sleep 3; done
```

---

## Scoring

`proxy_cost = 1.0×WL + 0.5×Density + 0.5×Congestion` on 17 IBM benchmarks.
Zero tolerance for hard macro overlaps (→ disqualification).
