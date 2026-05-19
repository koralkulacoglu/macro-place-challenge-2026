# HANDOFF: Macro Placement Challenge 2026

**Deadline: May 21, 2026 | Prize: $49K | Submission: https://forms.gle/YDRtYV5Vq68SZgKW9**

---

## Current Status (2026-05-19, end of day)

**Pipeline**: CT positions → Xplace multi-seed GP → legalize → sequential SA

**Git head**: `f7acf29` (main)

**Verified ibm01 numbers (measured today in Docker):**

| Stage | Proxy | Notes |
|-------|-------|-------|
| CT initial | ~1.04 | Before any optimization |
| Xplace GP (best of 6 seeds, ~36s) | 0.908 | After legalization, before SA |
| Xplace GP + full SA (3480s) | ~0.89 | Estimated from prior runs |

**Estimated average across all 17 benchmarks:**
- Xplace GP avg: ~1.25 (from prior per-benchmark measurements, see below)
- After SA: ~1.10–1.20 (SA gives ~8% improvement on average)
- Leaderboard rank 1 (Carrotato): 0.9671 average

**We are behind primarily on congestion.** Xplace GP optimizes WL + Density but NOT congestion. Congestion is ~40% of proxy cost for dense IBM benchmarks (cong ≈ 1.18–1.20 for ibm01 even after GP). This is the main gap.

---

## Leaderboard (May 18, 2026)

| Rank | Team | Avg Score | Method |
|------|------|-----------|--------|
| 1 | Carrotato | **0.9671** | Triton + Xplace + congestion-aware GP |
| 2 | Shoom | 0.978 | MultiDREAMPlace + CD |
| 3 | vmallela | 1.0109 | Fast evaluator + good GP |
| 5 | Cezar | 1.037 | Custom refinement |
| ~15–25 | **Us (estimated)** | ~1.10–1.20 | Xplace + SA |

---

## Verified Xplace GP Scores (before SA)

Measured across multiple seeds, best result shown (post-legalization):

| Benchmark | CT baseline | Xplace GP best | Δ |
|-----------|------------|----------------|---|
| ibm01 | 1.039 | **0.908** | −12.6% |
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

**Avg Xplace GP: ~1.25** (ibm02 and ibm18 are outliers — barely improved, unclear why)

---

## Current Pipeline Details

```
CT positions (benchmark.macro_positions from initial.plc)
    ↓
Xplace multi-seed GP (bookshelf format, ~700s budget)
    - 6 noise ratios: [0.005, 0.025, 0.05, 0.10, 0.15, 0.20]
    - stop_overflow=0.05, inner_iter=8000, mixed_size=True
    - Each seed ~6s on RTX 6000; ibm01 converges naturally to overflow≈0.05
    - Best seed by oracle cost (proxy after legalization) selected
    - ~16 top seeds kept for SA starting points
    ↓
_legalize_hard (greedy micro-legalization)
    ↓
Sequential SA (3480s budget)
    - CD → LNS → FD (congestion-guided gradient) → hSA → oSA
    - n_workers=1 (parallel SA deadlocks due to CUDA+fork)
    - FastEvaluator: 383x faster than oracle for SA accept/reject
    ↓
Final result
```

**Key parameters locked in:**
- `stop_overflow=0.05` — 0.01 crashes Xplace's internal LP macro legalization (PulpError with NaN/inf when cells are too tightly packed)
- `use_route_force=False` — route_force is infeasible (see below)
- `n_workers=1` — parallel SA deadlocks when CUDA is active (CUDA+fork futex issue)

---

## What We've Tried and Why It Didn't Work

### Xplace route_force (spent 2 full sessions investigating)

**Goal**: Enable `--use_route_force True` to add RUDY congestion gradients to Xplace ePlace.

**Attempt 1** (binary patching, bookshelf format):
- Root cause: `gpugr.GRDatabase` at offset 0x44806 explicitly checks `setting.Format != "lefdef"` and terminates. Bookshelf is rejected unconditionally.
- Tried binary patching to make ispd2005 call `readBSRoute` — works for file reading but GRDatabase still rejects it.

**Attempt 2** (synthetic LEF/DEF generation):
- Generated valid LEF/DEF files from bookshelf data (183 LEF layers = 37H + 55V + 91 CUT).
- `gpdb.setup()` succeeds! Core info correct: x=[0,22950] y=[0,23002].
- GP without route_force works fine (proxy=1.1965 for ibm01).
- GP with route_force=True: crashes SIGSEGV during GPU pattern routing.
- Root cause: IBM ICCAD04 benchmarks generate ~1.8M routing segments on the GPU's 32×32 routing grid. This overflows a hard-coded buffer in the CUDA pattern routing kernel. Not fixable without patching the CUDA binary.
- The fundamental conflict: IBM routing density (65–107 routes/µm) makes routing grid either too large (OOM) or too capacity-short (buffer overflow). No configuration avoids both.

**Conclusion**: route_force is fundamentally incompatible with IBM ICCAD04 benchmarks in Xplace. Do NOT attempt again.

---

## The Core Remaining Problem: Congestion During GP

Xplace ePlace optimizes WL + Density only. Congestion (40% of proxy cost) is ignored during GP. After GP, our SA's FD step does gradient descent on top-K congested macros — but this is a local fix on top of an already-congested layout.

**ibm01 breakdown after GP (before SA):**
- proxy = 0.908
- wl = 0.069 (7.6% of proxy)
- density = 0.500 (27.5% — ePlace does well here)
- congestion = 1.200 (66% — huge, entirely unoptimized)

Reducing congestion from 1.20 → 0.80 would drop total proxy by 0.50×0.20 = 0.10 → proxy ≈ 0.81. That would put us near rank 1.

---

## Research: Alternative GP Methods for Full Proxy Optimization

The goal: a GP stage that can directly reduce congestion cost while staying legal (no hard macro overlaps).

### Option 1: Custom RUDY Gradient Descent (RECOMMENDED — implement first)

**What it is**: After Xplace GP (which gives good WL+Density), run our own RUDY-based gradient descent to reduce congestion. No external tools needed.

**How RUDY works** (Rectangular Uniform wire Density):
For each net with bounding box W×H:
```
for each gcell (i,j) overlapping the bbox:
    h_demand[i,j] += 1/(W×H) × gcell_h × net_weight
    v_demand[i,j] += 1/(W×H) × gcell_w × net_weight
congestion = max(h_demand/h_cap, v_demand/v_cap)
```
Where `h_cap = hroutes_per_micron × gcell_h`, `v_cap = vroutes_per_micron × gcell_w`.

**Implementation plan:**
```python
def rudy_descent(positions, benchmark, n_steps=50, lr=0.05, plc=None):
    """Post-GP RUDY gradient descent to reduce congestion."""
    # 1. Compute RUDY demand map using torch (differentiable)
    # 2. Identify top-K congested cells
    # 3. For macros overlapping congested cells, compute ∂congestion/∂x via FD
    # 4. Step in anti-gradient direction (move away from congestion)
    # 5. Clamp to canvas, re-legalize
    # 6. Repeat
```

**Key advantage**: We already have `plc.get_horizontal_routing_congestion()` and `plc.get_grid_cell_of_node()`. Our SA's FD step already does a version of this. The new step would be more aggressive (dedicated phase, not mixed with other moves).

**Expected impact**: If we can reduce cong from 1.20 → 0.90 for ibm01, proxy drops from 0.908 → 0.758. Optimistic but directionally correct.

**Implementation difficulty**: Medium (1–2 days). The RUDY computation in PyTorch is straightforward. The challenge is keeping placements legal after gradient moves.

### Option 2: DREAMPlace with Routability Mode

**What it is**: DREAMPlace has `--routability_opt_flag 1` that adds RUDY-based congestion to the ePlace objective. It works with bookshelf format natively (no LEF/DEF conversion needed).

**What we know**:
- Old DREAMPlace (pre-entropy-injection): gave ibm01 = 0.9197 reliably (WL+Density only)
- Entropy injection bug: introduced in a newer commit, breaks convergence. Fix = pin to older commit.
- With routability mode: untested but should reduce congestion during GP

**How to fix entropy injection bug** (from memory):
```python
# In NonLinearPlace.py, the entropy injection fires when:
# overflow > 0.95 — which is ALWAYS true at center-init start
# Fix: patch to only fire after first 100 iterations, or raise threshold to 0.99
```

**Why entropy injection fires**: DREAMPlace starts all macros at center → overflow ≈ 1.0 → entropy fires immediately, disrupting optimization. The fix: increase the overflow threshold or delay the entropy injection.

**Implementation plan**:
1. Find the specific commit that introduced entropy injection in DREAMPlace repo
2. Pin Dockerfile to that commit (or apply patch)
3. Enable `--routability_opt_flag 1` in the DREAMPlace call
4. Compare against Xplace GP

**Expected impact**: If routability DREAMPlace reduces cong from 1.20 → 0.95, proxy ≈ 0.85 for ibm01.

**Implementation difficulty**: Medium (1 day). Main challenge is finding the right DREAMPlace commit and building the Docker image.

### Option 3: OpenROAD Global Placement (RePlAce)

**What it is**: OpenROAD's `replace` module — the same algorithm the competition uses as its ~1.46 baseline, but with routability-aware mode which should give much better results.

**Why it might help**: OpenROAD's RePlAce supports:
- `set_routing_driven 1` — enables RUDY-based routability
- Works with LEF/DEF input
- Mature, well-tested, used by many teams

**The LEF/DEF conversion problem**: We spent a full session generating synthetic LEF/DEF from bookshelf. The database loads correctly (Core coords right), but Xplace's route_force crashed. OpenROAD might handle the same LEF/DEF better — it's a different parser.

**Implementation plan**:
1. Add OpenROAD to Dockerfile (`apt-get install openroad` or build from source)
2. Reuse the `bookshelf_to_lefdef.py` converter we built (generates valid LEF/DEF)
3. Run: `openroad -exit < run_replace.tcl` with:
   ```tcl
   read_lef design.lef
   read_def design.def
   global_placement -routability_driven -density 0.95
   write_def output.def
   ```
4. Parse output DEF to get positions

**Key uncertainty**: Whether our synthetic LEF/DEF is valid enough for OpenROAD's parser. The LEF we generate has 183 layers (37H + 55V) which is unusual. May need simplification.

**Implementation difficulty**: High (2+ days). Docker build complexity + format conversion + debugging.

### Option 4: Custom Differentiable ePlace in PyTorch

**What it is**: Implement our own ePlace optimizer that includes ALL three proxy cost terms in the gradient.

**The objective**:
```
minimize: α·HPWL + β·density_penalty + γ·RUDY_congestion
```

Where:
- `HPWL`: smooth log-sum-exp approximation of wirelength
- `density_penalty`: Gaussian-smoothed density overflow (ePlace standard)
- `RUDY_congestion`: differentiable RUDY computation

**Why this is the cleanest solution**: We control the exact objective function. We can weight congestion as heavily as needed.

**Implementation sketch**:
```python
class CustomGP:
    def __init__(self, benchmark, wl_weight=1.0, density_weight=0.5, rudy_weight=0.5):
        ...
    
    def hpwl_loss(self, positions):
        # log-sum-exp smooth HPWL
        ...
    
    def density_loss(self, positions):
        # Gaussian KDE density penalty
        ...
    
    def rudy_loss(self, positions):
        # RUDY congestion from net bboxes
        ...
    
    def optimize(self, n_iters=5000, lr=0.01):
        optimizer = torch.optim.Adam([positions], lr=lr)
        for _ in range(n_iters):
            loss = self.hpwl_loss() + density_loss() + rudy_loss()
            loss.backward()
            optimizer.step()
```

**Key challenge**: Implementing the density penalty efficiently (ePlace uses FFT for the Poisson solver). A simplified bin-based density is feasible in PyTorch without FFT.

**Implementation difficulty**: Very high (3+ days). Not recommended given 2 days remaining.

### Option 5: Tuning Post-GP SA for Congestion

Our SA already has congestion-aware features. These could be strengthened:

1. **More aggressive FD phase**: 
   - Currently FD runs briefly before hSA
   - Could run 100–200 gradient steps instead of the current brief pass
   - Uses `plc.get_horizontal/vertical_routing_congestion()` for the gradient

2. **Dedicated congestion-reduction phase**:
   - Add a new phase BEFORE CD: run 50 FD steps targeting only congestion
   - Accept any move that reduces congestion (not full proxy)
   - This is essentially Option 1 but implemented inside SA

3. **Higher FD weight on congestion**:
   - Currently FD uses full proxy gradient
   - Could weight congestion term 2–3× higher to prioritize it

**Implementation difficulty**: Low (4 hours). Risk: might hurt WL quality while chasing congestion.

---

## Recommended Next Session Plan

### Day 1 (today — May 20): Custom RUDY gradient descent

**Goal**: Implement a post-GP congestion reduction step. This is the lowest-risk, highest-leverage option.

**Implementation** (in `placer.py`, ~100 lines):

```python
def _rudy_gradient_descent(self, positions, benchmark, plc, n_steps=80, probe=0.02):
    """Run RUDY-guided FD gradient descent to reduce congestion post-GP."""
    import numpy as np
    
    pos = positions.numpy().copy()
    sizes = benchmark.macro_sizes.numpy()
    n = benchmark.num_hard_macros
    movable = [i for i in range(n) if not benchmark.macro_fixed[i]]
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    
    # Get plc node IDs for hard macros
    plc_ids = benchmark.hard_macro_indices
    
    for step in range(n_steps):
        # Refresh congestion every 10 steps
        if step % 10 == 0:
            from macro_place.objective import compute_proxy_cost
            r = compute_proxy_cost(torch.tensor(pos, dtype=torch.float32), benchmark, plc)
            _h = np.array(plc.get_horizontal_routing_congestion(), dtype=np.float32)
            _v = np.array(plc.get_vertical_routing_congestion(), dtype=np.float32)
            _cong = _h + _v
            # Map each hard macro to its congestion cell
            _cells = {pid: plc.get_grid_cell_of_node(pid) for pid in plc_ids}
            _mc = np.array([_cong[_cells[plc_ids[i]]] if i < len(plc_ids) else 0 for i in movable])
        
        # Pick the most congested macro
        worst = movable[np.argmax(_mc)]
        
        # FD gradient: probe in ±x, ±y
        dx = cw * probe; dy = ch * probe
        for axis, delta in [(0, dx), (0, -dx), (1, dy), (1, -dy)]:
            old = pos[worst, axis]
            pos[worst, axis] = np.clip(old + delta, 
                sizes[worst, axis]/2, 
                (cw if axis==0 else ch) - sizes[worst, axis]/2)
            # Evaluate congestion at new position (fast, no SA overhead)
            # Accept if congestion decreases
            ...
    return torch.tensor(pos, dtype=torch.float32)
```

Insert this call between Xplace GP and SA in `_run_xplace_multiseed`.

**Test**: Does ibm01 proxy drop from 0.908 → 0.85? Check cong_cost specifically.

### Day 2 (May 21): DREAMPlace routability (if Day 1 shows promise)

If custom RUDY shows 5%+ congestion reduction, also try:
1. Pin DREAMPlace to pre-entropy-injection commit (see DREAMPlace GitHub history)
2. Enable `--routability_opt_flag 1`
3. Test ibm01: expect cong_cost < 1.0 from GP alone

---

## Key Technical Notes for Next Session

### Docker Usage (critical — bind mount doesn't work for new files)

The `koral-placer-xplace` image COPIES `placer.py`, `bookshelf.py`, `fast_eval.py` at build time. New files MUST be copied with `docker cp` into a running container, or the Dockerfile must be updated and image rebuilt.

```bash
# Start container with long sleep
docker run -d --runtime=nvidia --gpus all \
  -v "/c/path/to/repo:/challenge" --network none \
  --name test_container --entrypoint=bash koral-placer-xplace -c "sleep 3600"

# Copy new/updated files
docker cp submissions/koral/placer.py test_container:/challenge/submissions/koral/placer.py
docker cp submissions/koral/new_file.py test_container:/challenge/submissions/koral/new_file.py

# Run test
docker exec test_container bash -c "cd /challenge && python3 -m macro_place.evaluate submissions/koral/placer.py -b ibm01"
```

### Xplace parameters (locked in — do not change)

```python
--inner_iter 8000       # 8000 Nesterov iterations
--stop_overflow 0.05    # DO NOT lower (0.01 crashes LP macro legalization)
--mixed_size True       # Required for hard+soft macro handling
--use_route_force False # DO NOT enable (crashes GPU pattern router)
--noise_ratio varies    # [0.005, 0.025, 0.05, 0.10, 0.15, 0.20] per seed
--seed varies           # seed_42 through seed_47 (6 seeds per benchmark)
```

### SA structure (do not modify without testing)

The SA runs in sequential mode (n_workers=1) due to CUDA+fork deadlock. Total budget 3480s divided as:
- CD (coordinate descent): ~50 iterations
- LNS (large neighborhood search): 15–50% of budget
- FD (finite-difference gradient): brief, targeting top-K congested macros
- hSA (HPWL surrogate SA): bulk of remaining time
- oSA (oracle SA): tail phase

FastEvaluator (fast_eval.py) gives 383x speedup for SA accept/reject decisions.

### Key metrics for ibm01

| Metric | CT | Xplace GP | Target (rank 1 avg) |
|--------|----|-----------|--------------------|
| proxy | 1.04 | 0.908 | 0.97 (avg, not ibm01) |
| wl | 0.070 | 0.069 | ~0.05 |
| density | 0.812 | 0.500 | ~0.40 |
| congestion | 1.141 | 1.200 | ~0.80 |

Congestion is WORSE after Xplace GP than CT. This confirms Xplace ignores congestion entirely.

---

## Files

| File | Purpose | Status |
|------|---------|--------|
| `submissions/koral/placer.py` | `KoralPlacer` main (~1300 lines) | Current |
| `submissions/koral/bookshelf.py` | Benchmark → ISPD2005 bookshelf for Xplace | Current |
| `submissions/koral/fast_eval.py` | FastEvaluator (383x SA speedup) | Current |
| `submissions/koral/Dockerfile` | Docker: Xplace + dependencies | Current |
| `HANDOFF.md` | This file | Updated 2026-05-19 |

DREAMPlace, patch_dreamplace.sh, and related files were removed from the pipeline. The Dockerfile no longer builds DREAMPlace.

---

## Submission

When ready: https://forms.gle/YDRtYV5Vq68SZgKW9

Run all 17 benchmarks in Docker:
```bash
docker run --rm --runtime=nvidia --gpus all \
  -v "C:/Users/kulac/Documents/GitHub/macro-place-challenge-2026:/challenge" \
  --network none koral-placer-xplace \
  submissions/koral/placer.py --all
```
