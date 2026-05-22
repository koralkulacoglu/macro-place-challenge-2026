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
- Full `--all` baseline (GraphGradPlacer, early version): **1.2240 avg**

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

## Our Submission: `submissions/idk/`

**`GraphGradPlacer`** — Two-stage placer: soft-macro analytical GP (Adam optimizer, TILOS-faithful surrogate) followed by LAHC deep polish. Pure PyTorch + NumPy, no external tools, no Docker rebuild required.

**Pipeline:**

```
[Stage 1  Soft-Only Analytical GP]  ~5 min (gp_max_budget_s=300s)
   96 parallel soft-macro layouts optimized via Adam
   Hard macros locked at legalized initial.plc positions
   TILOS-faithful surrogate: exact density port (top-10% mean),
   WAHPWL normalized like the oracle, RUDY congestion with
   TILOS per-axis track normalization + 1D smoothing kernel
   Top-24 candidates scored by oracle; best zero-overlap result kept
       |
[Stage 2  FastEvaluator + LAHC]  ~49 min (lahc_min_budget_s=2970s)
   Bit-exact NumPy mirror of PlacementCost (~2000x faster per move)
   Late Acceptance Hill Climbing: L=100 history list
   Mixed moves: hard macro slide/swap + centroid-biased soft macro translate
   Best-ever placement tracked; oracle-verified before commit
```

**Why this design:**
- **TILOS-faithful surrogate.** The GP surrogate is an exact differentiable port of the judging metric — same density formula, same WL normalization, same congestion smoothing — so gradients point in the direction that actually lowers the score.
- **FastEvaluator speedup.** ~2000x faster than the oracle per incremental move enables ~1.5M LAHC iterations in the remaining budget.
- **Soft-only GP avoids legalization thrash.** Locking hard macros during GP means no expensive re-legalization between Adam steps; LAHC then freely moves both hard and soft macros.

**Key files:**
- `submissions/idk/placer.py` — `GraphGradPlacer` main class + `FastEvaluator` + `lahc_polish` (~2350 lines)
- `submissions/idk/gp.py` — older focused electrostatic GP module (not actively used)

**Key parameters (in `GraphGradPlacer.__init__`):**
- `gp_max_budget_s=300` — hard cap on the GP stage (5 min)
- `lahc_min_budget_s=2970` — LAHC floor (~49.5 min)
- `pop_size=96` — parallel soft-macro replicas
- `gp_k_eval=24` — top-K candidates scored by oracle after Adam

**Dev loop (Docker — no build needed, uses standard pytorch image):**
```bash
# Single benchmark with visualization
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none --entrypoint bash koral-placer-xplace \
  -c "cd /challenge && python3 -m macro_place.evaluate submissions/idk/placer.py -b ibm01 --vis"

# Full submission run
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none --entrypoint bash koral-placer-xplace \
  -c "cd /challenge && python3 -m macro_place.evaluate submissions/idk/placer.py --all"
```

`placer.py` is bind-mounted live. The `koral-placer-xplace` image is reused only for its PyTorch environment.

## Reproducibility Warning

Most competitive placers use non-deterministic algorithms (SA, GPU ops). Self-reported scores often differ from verified scores due to hardware differences and floating point non-associativity. To minimize the gap: fix random seeds, test inside Docker with the same PyTorch version as the judging image before submitting.
