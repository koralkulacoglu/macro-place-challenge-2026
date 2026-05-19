# HANDOFF: Macro Placement Challenge 2026

**Deadline: May 21, 2026 | Prize: $49K | Submission: https://forms.gle/YDRtYV5Vq68SZgKW9**

## Current Status (2026-05-19 Session 4)

- **Branch**: `strategy-may18` (now main)
- **Key fixes applied**: multi-fidelity Xplace seed search + oSA oracle sync + Adam WL+density + Phase A timeout fix
- **10-min Docker test results**: ibm06=1.3871 (was 1.4099 ✅), ibm13 started from 1.0604 (was 1.068 ✅), ibm01 Phase A timeout bug fixed
- **SA now contributes**: oSA oracle sync confirmed working (ibm01: 1.0315 → 1.0258, still improving)
- **Next**: Full 60-min runs on all 17 benchmarks once ready to submit

---

## Leaderboard (May 18, 2026)

| Rank | Team | Score | Key differentiator |
|------|------|-------|--------------------|
| 1 | Carrotato | 0.9671 | Triton + **Xplace** |
| 2 | Shoom | 0.978 | MultiDREAMPlace + CD |
| 3 | vmallela | 1.0109 | Fast evaluator + good GP |
| 5 | Cezar | 1.037 | Custom refinement |
| ~35 | **Us (before today)** | ~1.42–1.46 | CT + SA |
| **Us (with Xplace GP)** | **~1.08–1.20 est.** | Xplace + SA (SA results pending) |

---

## Verified Xplace GP Scores (before SA polish)

| Benchmark | CT baseline | Xplace GP | Improvement |
|-----------|------------|-----------|-------------|
| ibm01 | 1.039 | **0.910** | -12.4% |
| ibm02 | 1.566 | **1.552** | -0.9% |
| ibm03 | 1.326 | **1.176** | -11.3% |
| ibm04 | 1.313 | **1.115** | -15.1% |
| ibm06 | 1.658 | **1.410** | -15.0% |
| ibm07 | 1.476 | **1.270** | -14.0% |
| ibm08 | 1.466 | **1.339** | -8.7% |
| ibm09 | 1.113 | **0.952** | -14.5% |
| ibm10 | 1.340 | **1.135** | -15.3% |
| ibm11 | 1.214 | **0.979** | -19.3% |
| ibm12 | 1.625 | **1.369** | -15.8% |
| ibm13 | 1.385 | **1.068** | -22.9% |
| ibm14 | 1.594 | **1.392** | -12.7% |
| ibm15 | 1.603 | **1.517** | -5.4% |
| ibm16 | 1.491 | **1.325** | -11.1% |
| ibm17 | 1.739 | **1.469** | -15.5% |
| ibm18 | 1.790 | **1.764** | -1.5% |

**Average Xplace GP**: ~1.25 (vs CT ~1.46, -14% average)
**With SA improvement (~8%)**: **estimated ~1.15 average**

---

## Pipeline (Current)

```
CT positions (1.04–1.87 per benchmark)
    ↓ Try Xplace (routability-driven GP, requires Docker GPU) → 0.91–1.76
    ↓ Try DREAMPlace (patched entropy injection) → sometimes beats CT
    ↓ Best of (Xplace, DREAMPlace, CT)
    ↓
Sequential SA (3480s, GPU → n_workers=1 to avoid CUDA+fork deadlock)
    ↓ CD → LNS → FD → hSA → fast oSA
    ↓ FastEvaluator (383-17000x faster, wired into oSA)
    ↓
Final micro-legalize
```

---

## Key Technical Discoveries (Session 3)

### 1. FastEvaluator — DONE
- 383x faster than oracle on ibm01, 763x on ibm02
- 4061x faster delta_wl (incremental WL-only)
- Congestion map API for gradient-guided oSA moves
- r=0.98+ calibration vs official oracle

### 2. DREAMPlace Entropy Injection Fix — DONE
- Root cause: entropy injection fires at overflow > 0.95 (always for center-init)
- Fix: runtime-patch NonLinearPlace.py at startup → ibm01 gives 1.02 reliably

### 3. Xplace Integration — DONE (koral-placer-xplace image)
- Needs: CUDA, /opt/xplace, thirdparty/flute/*.dat, NumPy 2.0 patches
- ibm01 Xplace GP = 0.9099 (density drops 0.812→0.500, -38%)
- Best benchmarks: ibm11 (-19%), ibm13 (-23%)
- Weak benchmarks: ibm02 (-0.9%), ibm18 (-1.5%)
- **NOTE**: requires `--mixed_size True` for macro placement mode

### 4. CUDA+fork Deadlock Fix — DONE
- Problem: fork() after CUDA was active leaves all workers in futex_wait_queue
- Workers use ~0 CPU (deadlocked, not computing)
- Fix: when `torch.cuda.is_available()`, use n_workers=1 (sequential SA)
- Sequential SA with full budget from 0.91 starting point is highly effective

---

## Docker Commands

### Run single benchmark (correct command):
```bash
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none --entrypoint python3 koral-placer-xplace \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01 \
  > C:/tmp/final_ibm01.log 2>&1 &
```

### Run all 17 (for submission):
```bash
for bm in ibm01 ibm02 ibm03 ibm04 ibm06 ibm07 ibm08 ibm09 ibm10 ibm11 ibm12 ibm13 ibm14 ibm15 ibm16 ibm17 ibm18; do
  docker run --rm --runtime=nvidia --gpus all \
    -v "C:/path/to/repo:/challenge" \
    --network none --entrypoint python3 koral-placer-xplace \
    -m macro_place.evaluate submissions/koral/placer.py -b $bm \
    > C:/tmp/final_${bm}.log 2>&1 &
done
```

### Key Docker image:
- `koral-placer-xplace` = DREAMPlace + Xplace + all patches
- **Use `koral-placer-xplace`, NOT `koral-placer`** (old image, no Xplace)

### WSL2 crash workaround:
Docker Desktop WSL2 crashes during `docker build` with heavy CUDA compilation.
Use `docker exec` in a running container instead:
```bash
docker run -d --name build_container --entrypoint sleep koral-placer 7200
docker exec -d build_container bash -c "make -j\$(nproc) > /tmp/build.log 2>&1 && make install && echo DONE >> /tmp/build.log"
docker commit build_container koral-placer-xplace
```

---

## Active Runs (as of end of May 18 session)

| Benchmark | Log | Status |
|-----------|-----|--------|
| ibm01 | C:/tmp/final_ibm01.log | Sequential SA from Xplace 0.91 |
| ibm09 | C:/tmp/final_ibm09.log | Sequential SA from Xplace 0.95 |
| ibm06 | C:/tmp/final_ibm06.log | Sequential SA from Xplace 1.41 |
| ibm03 | C:/tmp/final_ibm03.log | Sequential SA |
| ibm04 | C:/tmp/final_ibm04.log | Sequential SA |

### Expected results (collect tomorrow May 19):
- ibm01: Xplace 0.91 + SA → expect ~0.82-0.88
- ibm09: Xplace 0.95 + SA → expect ~0.86-0.91
- ibm06: Xplace 1.41 + SA → expect ~1.28-1.35

---

## Key Files

| File | Purpose |
|------|---------|
| `submissions/koral/placer.py` | KoralPlacer: CT → Xplace → DREAMPlace → SA |
| `submissions/koral/fast_eval.py` | FastEvaluator: 383-17000x speedup |
| `submissions/koral/bookshelf.py` | Benchmark → Bookshelf (for Xplace/DREAMPlace) |
| `submissions/koral/Dockerfile` | Docker build: DREAMPlace + Xplace + patches |

---

## Next Steps (May 19-21)

### Priority 1: Collect results and submit
1. Check final scores from active runs (~1 hour each)
2. Run remaining benchmarks: ibm02, ibm07, ibm08, ibm10-18 in Docker
3. Submit when full set complete: https://forms.gle/YDRtYV5Vq68SZgKW9

### Priority 2: Improve Xplace quality (if time allows)
- Try `--use_route_force True` to directly target congestion (currently disabled)
- Try more iterations: `--inner_iter 8000` (currently 5000)
- ibm02/ibm18 barely improved → investigate why (likely Xplace parameter sensitivity)

### Priority 3: Better legalization
- After Xplace GP, many benchmarks have macros needing legalization
- Current `_legalize_hard` is greedy O(n²) Python — might create suboptimal arrangements
- Could try scipy-based legalization for better quality

### What NOT to do:
- Don't try to rebuild the Dockerfile from scratch (WSL2 crashes) — use docker exec approach
- Don't add more Python SA improvements (FastEvaluator + oSA already near saturation)
- Don't worry about parallel SA — sequential SA from Xplace is sufficient
