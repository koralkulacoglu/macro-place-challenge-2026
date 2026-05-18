# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Repo Is

A competition scaffold for the **Partcl/HRT Macro Placement Challenge** (deadline May 21, 2026, submission link: https://forms.gle/YDRtYV5Vq68SZgKW9). The goal is to write the best algorithm for placing hard macros (SRAMs, IPs, etc.) on a chip floorplan. Prize pool: $49,000.

The problem is NP-hard — both the feasibility subproblem (rectangle packing without overlap) and the optimization objective (wirelength minimization, a variant of QAP) are independently NP-hard.

## Commands

```bash
# Setup (one-time)
git submodule update --init external/MacroPlacement
uv sync

# Run your placer on a single benchmark
uv run evaluate submissions/your_placer.py -b ibm01

# Run on all 17 IBM benchmarks (what gets scored)
uv run evaluate submissions/your_placer.py --all

# Run on NG45 commercial designs (Tier 2 judging designs)
uv run evaluate submissions/your_placer.py --ng45

# Visualize — saves to vis/<benchmark>.png
uv run evaluate submissions/your_placer.py --all --vis

# Tests (require submodule initialized)
uv run pytest
uv run pytest test/test_smoke.py::test_greedy_row_placer
```

## Scoring

**Tier 1 (all submissions):** Ranked by average proxy cost across 17 IBM benchmarks. Lower is better.
```
Proxy Cost = 1.0 × Wirelength + 0.5 × Density + 0.5 × Congestion
```
Any benchmark with a hard macro overlap → disqualified entirely.

**Tier 2 (top 7 only):** Judges run the full OpenROAD PnR flow on NG45 designs (ariane133, ariane136, mempool_tile, nvdla + 1-2 hidden). Measures WNS, TNS, Area. Grand Prize uses a weighted geometric mean score with weights WNS:3, TNS:2, Area:1.

**Key baselines to beat:**
- SA baseline avg: ~2.13
- RePlAce baseline avg: ~1.46 (the real target — harder to beat)
- Leaderboard rank 1: 0.9671 (Carrotato — Triton+Xplace)
- Leaderboard rank 5: 1.037 (Cezar)
- Leaderboard rank 10: 1.1764 (KLA MACH — DREAMPlace+CD+SA+ILS)
- Estimated our current: ~1.45–1.55 (~rank 35–45)

## Architecture

The framework is read-only — only add files under `submissions/`.

### Data flow

```
netlist.pb.txt + initial.plc
        ↓ loader.py
  (Benchmark, PlacementCost)
        ↓ your placer
  [num_macros, 2] tensor
        ↓ objective.py
  proxy cost dict
```

### Key objects

**`Benchmark`** ([macro_place/benchmark.py](macro_place/benchmark.py)) — pure PyTorch tensors:
- `macro_positions [N, 2]` — (x, y) center coords in microns, hard macros first (`[0, num_hard)`), soft macros after (`[num_hard, num_macros)`)
- `macro_sizes [N, 2]` — (width, height)
- `macro_fixed [N]` — bool mask; fixed macros must not move
- `net_nodes` — list of tensors, one per net, containing macro/port indices
- `net_pin_nodes` — pin-level connectivity: `[pins_in_net, 2]` where col 0 = owner index, col 1 = pin slot into `macro_pin_offsets`
- `macro_pin_offsets` — list of `[num_pins, 2]` tensors per hard macro (relative offsets from center)
- `port_positions [P, 2]` — I/O port positions on canvas boundary

**`PlacementCost`** ([macro_place/_plc.py](macro_place/_plc.py)) — TILOS evaluator object. Needed for cost computation. Exposes `optimize_stdcells()` for force-directed soft macro placement. Access nets via `plc.nets` (dict of driver→sinks by pin name).

**`compute_proxy_cost(placement, benchmark, plc)`** ([macro_place/objective.py](macro_place/objective.py)) — sets positions in PlacementCost, calls its cost methods, returns dict with `proxy_cost`, `wirelength_cost`, `density_cost`, `congestion_cost`, `overlap_count`, etc. Contains a monkey-patch fixing a boundary bug in the upstream `__get_grid_cell_location`.

### Writing a placer

Your submission is a `.py` file in `submissions/` with a class that has `place(self, benchmark: Benchmark) -> torch.Tensor`. The evaluate harness auto-discovers the first class with a `place` method.

```python
class MyPlacer:
    def place(self, benchmark: Benchmark) -> torch.Tensor:
        placement = benchmark.macro_positions.clone()
        # Hard macros: indices [0, num_hard_macros) — primary optimization target
        # Soft macros: indices [num_hard_macros, num_macros) — move these too for best results
        # Positions are CENTER coordinates, not corners
        # Fixed macros must stay put: benchmark.get_movable_mask()
        return placement
```

Constraints: center coordinates, no hard macro overlaps (zero tolerance), all macros within canvas bounds, fixed macros unmoved.

### Soft macros

Soft macros are standard cell cluster abstractions — they connect hard macros to each other and to I/O ports. Moving hard macros without repositioning soft macros degrades all three cost components. The SA baseline calls `plc.optimize_stdcells()` after each batch of hard macro moves (slow, ~minutes per call in Python). You can implement your own soft macro optimization on GPU instead.

### Net connectivity for custom algorithms

`benchmark.net_nodes` and `benchmark.net_pin_nodes` are the PyTorch-native way to access connectivity. For GNN-based approaches, build an edge index from `net_nodes`. For pin-level HPWL in a differentiable loss, use `net_pin_nodes` + `macro_pin_offsets` to get exact pin positions.

The `will_seed` placer ([submissions/will_seed/placer.py](submissions/will_seed/placer.py)) shows how to build a weighted edge dict from `plc.nets` for SA wirelength evaluation.

## Judging Environment

- Hardware: AMD EPYC 9655P, 16 cores, 100GB RAM, NVIDIA RTX 6000 Ada 48GB
- Base Docker image: `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime` (Python 3.11)
- Runtime: `--network none` — all deps must be installed at Docker build time
- Time limit: 1 hour per benchmark (17 hours total for `--all`)
- Judges run `uv run evaluate submissions/your_placer.py --all` (equivalent)

If you have non-standard deps, include a `Dockerfile` in your submission. Otherwise your `placer.py` is mounted into the judges' standard image.

## Our Submission: `submissions/koral/`

CT positions + parallel SA + ibm06 cascade fix. DREAMPlace is included but broken in the current Docker image — see HANDOFF.md for full details.

**Pipeline (git: 5a31474, full run in progress):**
1. CT positions → legalize → proxy eval (baseline ~1.04–1.57 per benchmark)
2. DREAMPlace center-init (stochastic; ~25% chance of beating CT; auto-discarded if worse)
3. Best of (CT, DREAMPlace) → 4 parallel SA workers (seeds 42–45)
   - Perturbation: 0%, 1.5%, 3%, 4.5% Gaussian noise (σ = fraction of canvas)
4. `_cd_lns_polish` (3480s budget per worker):
   - CD → LNS → FD → hSA (65% translate, 15% rotate, 20% swap)
   - Oracle SA tail: 70% translate, 30% swap (NEW: swap moves added May 18)
5. Best worker result used as final placement

**Key empirical findings (verified May 2026):**
- DREAMPlace (current Docker, May 2026 version): unreliable. center-init gives 1.01–1.23 (vs CT 1.04) stochastically. CT-init creates 139-200 hard macro overlaps → always discarded. OLD DREAMPlace gave ibm01=0.9197 reliably. FIX: pin Docker to pre-entropy-injection commit.
- ibm06: CT has 47 sub-4nm overlaps → cascade in micro_legalize. Perturbed workers at 3-4.5% sigma escape → 1.6877 (was 1.8343). This is the biggest SA improvement.
- Swap moves (30% in oracle SA): just added, unverified improvement. Should help WL-dominated.
- k_osa_max=1 is critical — cluster moves have <5% validity rate on dense benchmarks.
- LNS: 0.001–0.003 improvement for WL-dominated benchmarks.

**Best verified scores (May 2026):**
- ibm01: 0.9197 (old DREAMPlace, first session) / ~1.03 (current, CT+SA)
- ibm02: 1.5476 (CT+SA)
- ibm06: 1.6877 (perturbed SA cascade escape)
- ibm09: 1.1126
- Average (all 17, old baseline): 1.457
- Average (estimated, current): ~1.42–1.45

**Leaderboard position (estimated):** ~rank 30–40. Target: top 5 (need ~1.04).
**Critical gap:** analytical placement. Fix DREAMPlace version OR integrate Xplace.
See HANDOFF.md for all next steps.

**Key files:**
- `submissions/koral/placer.py` — `KoralPlacer` class (~1291 lines)
- `submissions/koral/bookshelf.py` — Benchmark → DREAMPlace Bookshelf adapter
- `submissions/koral/Dockerfile` — clones DREAMPlace at build time, applies CUDA 12.4 patches
- `submissions/koral/patch_dreamplace.sh` — CUDA 12.4 + NumPy 2.0 compat patches
- `HANDOFF.md` — full strategic context, leaderboard analysis, implementation plans

**Docker build:** Clones DREAMPlace from GitHub at build time. `--network none` is runtime only.

**CUDA 12.4 compatibility patches (applied by patch_dreamplace.sh at Docker build):**
- `cmake/TorchExtension.cmake`: detect CUDA via `torch.version.cuda` not `is_available()`
- `libcuda.so` stub symlinked for cmake's `find_package(CUDA)` during build
- 4 CUDA targets disabled: `pin_pos_cuda_segment`, `k_reorder_cuda`, `global_swap_cuda`, `independent_set_matching_cuda` (CUB API incompatible with CUDA 12.4; all are detailed-placement ops, unused with `detailed_place_flag=0`)
- `PlaceDB.py`: `np.string_` → `np.bytes_` (NumPy 2.0)

**Dev loop:**
```bash
# Build Docker image (clones + compiles DREAMPlace, ~40 min first time)
docker build -t koral-placer -f submissions/koral/Dockerfile .

# Run with live repo bind-mount + GPU (ibm01 needs DREAMPlace = needs Docker)
docker run --rm --runtime=nvidia --gpus all -v $(pwd):/challenge \
  --network none --entrypoint python koral-placer \
  -m macro_place.evaluate submissions/koral/placer.py -b ibm01

# ibm02-18: no DREAMPlace needed, run locally
nohup timeout 620 uv run evaluate submissions/koral/placer.py -b ibm02 > /tmp/ibm02.log 2>&1 &
```

## Reproducibility Warning

Most competitive placers use non-deterministic algorithms (SA, GPU ops). Self-reported scores often differ from verified scores due to hardware differences and floating point non-associativity. To minimize the gap: fix random seeds, test inside Docker with the same PyTorch version as the judging image before submitting.
