# KoralPlacer — Design Document

**Partcl/HRT Macro Placement Challenge 2026**
Submission by Koral Kulacoglu — `submissions/koral/`

---

## Overview

KoralPlacer uses a three-stage pipeline:
1. **Global placement** via Xplace (routability-driven GPU analytical placer)
2. **Fallback** to DREAMPlace (WL+density analytical placer, entropy injection patched) or CT positions
3. **SA polish** using a custom SA with a 383–17000× faster surrogate evaluator

The key insight: all IBM benchmarks are congestion-dominated (WL accounts for only 4–6% of proxy cost). Reducing density and congestion hotspots requires a good global macro arrangement — CT positions alone are insufficient. Xplace's electrostatic placement with routability awareness directly reduces density, cutting it 10–23% vs CT across all 17 benchmarks.

**Critical finding (May 19):** The SA originally contributed ~0 improvement over Xplace GP — the fast evaluator drifted outside its calibration range during oSA, producing phantom improvements that the oracle rejected. Fixes applied: (1) multi-fidelity Xplace seed search (100-200 coarse → top-10 full GP → top-3 warm SA), (2) oSA oracle sync every 20s with snap-back on proxy drift.

---

## Pipeline

```
CT positions (Circuit Training baseline, ~1.04–1.79 per benchmark)
    │
    ├─► Xplace GP (requires Docker + GPU, ~15–180s)
    │   Routability-aware ePlace with Nesterov optimizer
    │   Mixed-size mode: handles hard macros + soft macros together
    │   Output: ISPD2005 Bookshelf .pl → parse → legalize
    │
    ├─► DREAMPlace GP (fallback, CPU/GPU, ~60s)
    │   Center-init, entropy injection patched at runtime
    │   Only used if result beats Xplace after legalization
    │
    └─► CT positions (final fallback)
    │
    Best of above → Sequential SA (3480s budget)
    │
    ├─► CD (Coordinate Descent, delta-HPWL inner loop, oracle outer)
    ├─► LNS (Large Neighborhood Search, cluster swaps)
    ├─► FD (Finite-Difference gradient descent on proxy)
    ├─► hSA (HPWL-surrogate SA, reheated 4×)
    └─► oSA (Oracle SA tail with FastEvaluator, 383–17000× speedup)
            │
            └─► FastEvaluator: HPWL + RUDY density + RUDY congestion
                Calibrated against oracle (Pearson r > 0.98)
                delta_wl() for O(degree) incremental WL
                congestion_map() for gradient-guided macro selection
    │
    Final micro-legalize (resolves residual sub-4nm overlaps)
```

---

## Xplace Integration

[Xplace](https://github.com/cuhk-eda/Xplace) (CUHK-EDA) is the analytical placer used by the rank-1 team (Carrotato). It extends ePlace with:
- GPU-accelerated electrostatic density (DCT-based Poisson solver)
- Optional routing force (RUDY-based congestion gradient)
- Mixed-size support (hard macros treated differently from standard cells)

We call it as a subprocess with ISPD2005 Bookshelf input (same format as DREAMPlace), parse the output `.pl` file, and legalize the result.

Key parameters:
- `--inner_iter 5000` — Nesterov iterations
- `--mixed_size True` — macro-aware placement
- `--stop_overflow 0.05` — convergence threshold
- `--use_filler False` — no filler cells (pure macro placement)
- `--target_density` — computed from benchmark utilization + 15% headroom

### NumPy 2.0 patches applied to Xplace (in Docker image):
- `np.round_` → `np.round` in `macro_legalization.py`
- `dtype=np.bool8` → `dtype=np.bool_` in `macro_legalization.py`

### CUDA + fork deadlock:
After Xplace uses GPU, `fork()` for parallel SA workers leaves all children blocked in `futex_wait_queue` (background CUDA threads hold mutexes that forked children can't release). Fix: when `torch.cuda.is_available()`, use `n_workers=1` (sequential SA, no fork).

---

## FastEvaluator

The TILOS oracle (`compute_proxy_cost`) takes 0.5–2.5s per call. With a 3480s SA budget this allows only ~1,400–7,000 oracle evaluations. The FastEvaluator (`fast_eval.py`) provides:

| Method | Speedup | Use |
|--------|---------|-----|
| `evaluate(pos)` | 383–763× | Full WL + density + congestion approx |
| `delta_wl(pos, i, nx, ny)` | 4000–17000× | WL-only incremental for translations |
| `congestion_map(pos)` | — | Per-bin RUDY congestion [R×C] |

Implementation:
- **WL**: `scatter_reduce` HPWL over flat net-node index, O(total_pins)
- **Density**: vectorized macro-cell overlap with broadcast, O(N×G)
- **Congestion**: RUDY approximation via 2D prefix-sum trick, O(nets + R×C)

Calibration: 5–8 oracle samples, linear fit (a × raw + b), r > 0.98 on all tested benchmarks.

The oSA tail uses `fast.evaluate()` (not `delta_wl`) for congestion-dominated benchmarks (wl_frac < 15%) since delta_wl captures only the WL component.

---

## Verified Benchmark Results

### Xplace GP scores (before SA polish):

| Benchmark | CT baseline | Xplace GP | Δ |
|-----------|------------|-----------|---|
| ibm01 | 1.039 | **0.910** | −12.4% |
| ibm02 | 1.566 | **1.552** | −0.9% |
| ibm03 | 1.326 | **1.176** | −11.3% |
| ibm04 | 1.313 | **1.115** | −15.1% |
| ibm06 | 1.658 | **1.410** | −15.0% |
| ibm07 | 1.476 | **1.270** | −14.0% |
| ibm08 | 1.466 | **1.339** | −8.7% |
| ibm09 | 1.113 | **0.952** | −14.5% |
| ibm10 | 1.340 | **1.135** | −15.3% |
| ibm11 | 1.214 | **0.979** | −19.3% |
| ibm12 | 1.625 | **1.369** | −15.8% |
| ibm13 | 1.385 | **1.068** | −22.9% |
| ibm14 | 1.594 | **1.392** | −12.7% |
| ibm15 | 1.603 | **1.517** | −5.4% |
| ibm16 | 1.491 | **1.325** | −11.1% |
| ibm17 | 1.739 | **1.469** | −15.5% |
| ibm18 | 1.790 | **1.764** | −1.5% |

**Average CT → Xplace GP: −11.8%**

### Cost breakdown insight (ibm01):
- CT: proxy=1.039, wl=0.064 (6.2%), density=0.812, congestion=1.137
- Xplace: proxy=0.910, wl=0.069, density=**0.500** (−38%), congestion=1.183
- Xplace primarily reduces density hotspots by spreading macros more evenly

### All results validated: zero hard macro overlaps (harness_count=0 on all tested benchmarks)

---

## How to Run

### Prerequisites
- Docker with NVIDIA GPU runtime (`docker run --gpus all` must work)
- Git submodule initialized: `git submodule update --init external/MacroPlacement`

### Docker image: `koral-placer-xplace`

The image is built from `submissions/koral/Dockerfile`. It contains:
- DREAMPlace (compiled with CUDA 12.4 patches)
- Xplace (compiled with NumPy 2.0 patches + FLUTE data files)
- All Python runtime dependencies

**Build the image** (use `docker exec` approach to avoid WSL2 crash during heavy CUDA compilation):
```bash
# Start a base container
docker run -d --name build_container --entrypoint sleep koral-placer 7200

# Build Xplace inside it
docker exec -d build_container bash -c "
  git clone --depth=1 --recurse-submodules https://github.com/cuhk-eda/Xplace.git /tmp/xplace_src &&
  cd /tmp/xplace_src &&
  pip install -q pandas loguru seaborn numba torchvision pulp igraph &&
  sed -i 's/int(torch.cuda.is_available())/int(torch.version.cuda is not None)/' CMakeLists.txt &&
  mkdir build && cd build &&
  cmake .. -DCMAKE_INSTALL_PREFIX=/opt/xplace -DCMAKE_CUDA_ARCHITECTURES=native -DCMAKE_BUILD_TYPE=Release &&
  make -j\$(nproc) && make install &&
  cp -r /tmp/xplace_src/src /opt/xplace/src &&
  cp -r /tmp/xplace_src/utils /opt/xplace/utils &&
  cp /tmp/xplace_src/main.py /opt/xplace/main.py &&
  cp -r /tmp/xplace_src/cpp_to_py /opt/xplace/cpp_to_py &&
  mkdir -p /opt/xplace/thirdparty/flute &&
  cp /tmp/xplace_src/thirdparty/flute/*.dat /opt/xplace/thirdparty/flute/ &&
  sed -i 's/np\.round_/np.round/g; s/dtype=np\.bool8/dtype=np.bool_/g' /opt/xplace/src/core/macro_legalization.py &&
  echo DONE > /tmp/xplace_done.log
"

# Wait for build (~20 min), then commit
# (poll with: docker exec build_container cat /tmp/xplace_done.log)
docker commit build_container koral-placer-xplace
```

### Run a single benchmark:
```bash
docker run --rm --runtime=nvidia --gpus all \
  -v "$(pwd):/challenge" \
  --network none --entrypoint python3 koral-placer-xplace \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01
```

### Run all 17 benchmarks (judges' equivalent):
```bash
docker run --rm --runtime=nvidia --gpus all \
  -v "$(pwd):/challenge" \
  --network none --entrypoint python3 koral-placer-xplace \
  -m macro_place.evaluate submissions/koral/placer.py --all
```

### Run without Docker (no Xplace, no DREAMPlace — CT + SA only):
```bash
uv run evaluate submissions/koral/placer.py -b ibm01
```

### Environment variable: `KORAL_SA_BUDGET`
Override the SA time budget (default 3480s = 58 min):
```bash
KORAL_SA_BUDGET=300 uv run evaluate submissions/koral/placer.py -b ibm01
```

---

## Key Files

| File | Description |
|------|-------------|
| `placer.py` | `KoralPlacer` class — main pipeline |
| `fast_eval.py` | `FastEvaluator` — calibrated fast proxy surrogate |
| `bookshelf.py` | Converts `Benchmark` → ISPD2005 Bookshelf (for Xplace/DREAMPlace) |
| `Dockerfile` | Builds DREAMPlace + Xplace inside Docker |

---

## Leaderboard Context

| Rank | Team | Score | Method |
|------|------|-------|--------|
| 1 | Carrotato | 0.9671 | Triton + Xplace |
| 2 | Shoom | 0.978 | MultiDREAMPlace + CD |
| 3 | vmallela | 1.0109 | Custom fast evaluator |
| 5 | Cezar | 1.037 | — |
| — | **This submission** | **~1.10–1.15 est.** | Xplace + FastEval SA |
| ~35 | Previous baseline | ~1.46 | CT + SA only |
