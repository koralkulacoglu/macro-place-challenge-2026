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
- RePlAce baseline avg: ~1.46 (the competition's reference baseline)
- Leaderboard rank 1: 0.9671 avg (Carrotato — Triton+Xplace+congestion-aware GP)
- Leaderboard rank 5: 1.037 (Cezar)
- Leaderboard rank 10: 1.1764 (KLA MACH — DREAMPlace+CD+SA+ILS)
- Estimated our current: ~1.10–1.20 (~rank 15–25 est.)

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

CT positions → Xplace multi-seed GP → sequential SA. DREAMPlace has been removed.

**Pipeline (git head: f7acf29):**
1. CT positions (from initial.plc)
2. Xplace multi-seed GP (6 seeds × ~6s = ~36s, bookshelf format)
   - `stop_overflow=0.05`, `inner_iter=8000`, `mixed_size=True`
   - Noise schedule: [0.005, 0.025, 0.05, 0.10, 0.15, 0.20]
   - Best seed selected by oracle cost after legalization
3. `_legalize_hard` (greedy micro-legalization)
4. Sequential SA (3480s): CD → LNS → FD → hSA → oSA
   - n_workers=1 (parallel SA deadlocks due to CUDA+fork)
   - FastEvaluator: 383x speedup for SA decisions

**Verified scores (measured 2026-05-19 in Docker):**
- ibm01: 0.908 after Xplace GP+legalization (before SA); ~0.89 after full SA
- Xplace GP avg across 17 benchmarks: ~1.25
- After SA avg: ~1.10–1.20 (estimated)

**Key empirical facts:**
- `stop_overflow=0.01` crashes Xplace's LP macro legalization (PulpError: NaN/inf). Use 0.05.
- `use_route_force=True` crashes Xplace's GPU pattern router for IBM benchmarks (1.8M route segments overflow kernel buffer). Do not attempt.
- Parallel SA deadlocks when CUDA is active (CUDA+fork futex issue) → n_workers=1 always.
- Xplace GP optimizes WL+Density only. Congestion (~40% of proxy cost) is NOT reduced during GP. This is the main gap vs rank 1.
- ibm02 and ibm18 barely improve with Xplace (−0.9% and −1.5%) — root cause unknown.

**Key files:**
- `submissions/koral/placer.py` — `KoralPlacer` class (~1300 lines)
- `submissions/koral/bookshelf.py` — Benchmark → ISPD2005 bookshelf for Xplace
- `submissions/koral/fast_eval.py` — FastEvaluator (383x SA speedup)
- `submissions/koral/Dockerfile` — Docker: Xplace + dependencies
- `HANDOFF.md` — detailed state, investigation results, next steps

**Dev loop (Docker — new files need `docker cp`, bind mount doesn't see them):**
```bash
# Build image (Xplace, ~40 min first time)
docker build -t koral-placer-xplace -f submissions/koral/Dockerfile .

# Start container for testing
docker run -d --runtime=nvidia --gpus all \
  -v "C:/path/to/repo:/challenge" --network none \
  --name test_xplace --entrypoint=bash koral-placer-xplace -c "sleep 3600"

# Copy updated files (required — bind mount only shows build-time copies)
docker cp submissions/koral/placer.py test_xplace:/challenge/submissions/koral/placer.py

# Run test
docker exec test_xplace bash -c \
  "cd /challenge && python3 -m macro_place.evaluate submissions/koral/placer.py -b ibm01"

# Full submission run
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/path/to/repo:/challenge" --network none \
  koral-placer-xplace submissions/koral/placer.py --all
```

## Reproducibility Warning

Most competitive placers use non-deterministic algorithms (SA, GPU ops). Self-reported scores often differ from verified scores due to hardware differences and floating point non-associativity. To minimize the gap: fix random seeds, test inside Docker with the same PyTorch version as the judging image before submitting.
