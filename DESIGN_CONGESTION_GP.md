# Design: Congestion-Aware GP via Xplace Seeds → DREAMPlace Routability

## Problem

Xplace ePlace optimizes WL + Density only. Congestion (~40% of proxy cost) is ignored
during GP and is actually **worse** after Xplace than CT for many benchmarks
(ibm01: CT cong=1.14 → Xplace cong=1.20). The SA's FD phase partially addresses this
post-hoc but can't escape the basin Xplace found.

The core issue is Xplace finds a WL+Density local minimum that happens to be
congested. We need a GP that has congestion in its objective.

## Proposed Pipeline

```
CT positions
    ↓
Xplace multi-seed GP (~400s, ~60 seeds)
    - Diverse noise ratios create diverse starting basins
    - Pick top-K seeds by oracle proxy cost (after legalization)
    ↓
DREAMPlace routability (top-K seeds, ~60-120s each)
    - Start from Xplace positions → overflow ≈ 0.05 (avoids entropy injection bug)
    - --routability_opt_flag 1 adds RUDY congestion penalty to ePlace objective
    - Runs until convergence from the already-good Xplace start
    - Pick best result by oracle proxy cost
    ↓
Sequential SA (remaining budget, ~2800-3000s)
    - CD → LNS → FD → hSA → oSA (unchanged)
```

## Why DREAMPlace Routability ≠ Xplace route_force

Xplace route_force uses a full GPU global router (gpugr.GRDatabase) which crashes
on IBM benchmarks with 1.8M route segments overflowing a CUDA kernel buffer.

DREAMPlace's `--routability_opt_flag 1` uses **analytical RUDY** — it computes routing
demand directly from net bounding boxes as a differentiable tensor operation, with no
GPU router. Formula:
```
for each net with bbox W×H:
    h_demand[cell] += 1/(W×H) × gcell_h    (for each cell overlapping bbox)
    v_demand[cell] += 1/(W×H) × gcell_w
congestion_penalty = sum(max(0, demand/capacity - 1)^2)
```
This is pure PyTorch math — no external router, no buffer limits, bookshelf-native.

## Why Starting from Xplace Positions Fixes the Entropy Injection Bug

DREAMPlace's entropy injection fires when `overflow > 0.95`. Starting from center
positions always triggers this (overlap from center init → overflow ≈ 1.0 immediately).

Starting from Xplace GP positions:
- overflow ≈ 0.05 (Xplace already spread the macros)
- Entropy injection never fires (threshold not reached)
- No patch needed — just use Xplace as the initializer

## Time Budget

Total budget: 3600s

| Phase | Time | Notes |
|-------|------|-------|
| Xplace multi-seed | ~400s | ~65 seeds × 6s each |
| Top-K selection | ~5s | compute oracle cost for top seeds |
| DREAMPlace routability (K=3) | ~180-360s | 60-120s per run |
| Sequential SA | ~2800-3000s | bulk of budget |

K=3 is a reasonable start. If DREAMPlace runs are fast (convergence from good init),
increase K. If slow, reduce to K=2 or K=1.

## Implementation Steps

### Step 1: Add DREAMPlace back to Dockerfile

DREAMPlace was removed in commit 377ce90. Add it back with CUDA 12.4 patches.

The patches (from `patch_dreamplace.sh`, applied at build time):
- `cmake/TorchExtension.cmake`: `torch.version.cuda` not `is_available()`
- `libcuda.so` stub for cmake find_package
- 4 CUDA targets disabled: `pin_pos_cuda_segment`, `k_reorder_cuda`,
  `global_swap_cuda`, `independent_set_matching_cuda`
- `PlaceDB.py`: `np.string_` → `np.bytes_`

The entropy injection bug: find the commit in DREAMPlace repo where `entropy_start_iter`
or similar was added, and pin to just before it. Or patch the condition:
```python
# In NonLinearPlace.py, find the entropy injection condition:
# Change: if overflow > 0.95:
# To:     if overflow > 0.95 and iteration > 100:
# This gives the optimizer time to spread before entropy fires
```

### Step 2: Write Xplace positions to .pl for DREAMPlace

After Xplace GP + legalization, write positions to bookshelf `.pl` format:

```python
def write_pl_from_positions(positions, benchmark, pl_path):
    """Write legalized positions back to bookshelf .pl for DREAMPlace init."""
    SCALE = 1000
    sizes = benchmark.macro_sizes.numpy()
    pos = positions.numpy()
    lines = ["UCLA pl 1.0", ""]
    for i in range(benchmark.num_macros):
        xl = int((pos[i,0] - sizes[i,0]/2) * SCALE)
        yl = int((pos[i,1] - sizes[i,1]/2) * SCALE)
        tag = " /FIXED" if bool(benchmark.macro_fixed[i]) else ""
        lines.append(f"n{i} {xl} {yl} : N{tag}")
    # Ports
    for j, pp in enumerate(benchmark.port_positions):
        px = int(float(pp[0]) * SCALE)
        py = int(float(pp[1]) * SCALE)
        lines.append(f"port{j} {px} {py} : N /FIXED_NI")
    with open(pl_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
```

### Step 3: Run DREAMPlace with routability

```python
def run_dreamplace_routability(benchmark, tmpdir, init_pl_path, seed=42):
    """Run DREAMPlace with routability from Xplace-initialized positions."""
    # Write bookshelf files (nodes, nets, scl, aux) — already done for Xplace
    # Override .pl with Xplace positions
    
    params = {
        # Input
        "aux_input": aux_path,
        "load_from_pl": init_pl_path,   # ← KEY: start from Xplace positions
        
        # Routability
        "routability_opt_flag": 1,
        "routability_opt_freq": 10,      # how often to recompute RUDY
        "route_weight": 0.1,             # congestion penalty weight
        
        # GP settings
        "target_density": target_density,
        "stop_overflow": 0.01,           # tighter than Xplace (starting from 0.05)
        "inner_iter": 1000,              # fewer iters — already converged on WL/density
        "gpu": 1,
        "num_threads": 8,
        "seed": seed,
        
        # Disable detail placement (we do our own SA)
        "detailed_place_flag": 0,
        "legalize_flag": 0,
        
        # Output
        "result_dir": os.path.join(tmpdir, f"dp_{seed}"),
        "output_file": "dreamplace_gp.pl",
    }
    # Run DREAMPlace subprocess
    ...
```

### Step 4: Select best result and pass to SA

```python
# After running K DREAMPlace seeds:
best_pos, best_cost = None, float('inf')
for dp_result in dreamplace_results:
    pos = parse_dreamplace_output(dp_result)
    pos = legalize_hard(pos, benchmark)
    if count_overlaps(pos) == 0:
        cost = compute_proxy_cost(pos, benchmark, plc)['proxy_cost']
        if cost < best_cost:
            best_cost, best_pos = cost, pos

# Fall back to best Xplace seed if DREAMPlace fails all K
if best_pos is None:
    best_pos = best_xplace_position
```

## Expected Results (Estimates)

| Stage | ibm01 proxy | Notes |
|-------|-------------|-------|
| Xplace GP (current) | 0.908 | cong=1.20 |
| + DREAMPlace routability | ~0.80–0.87 | cong target: 0.90–1.00 |
| + SA | ~0.75–0.82 | SA improvement ~8% |

Average across 17 benchmarks:
- Current Xplace GP avg: ~1.25
- With DREAMPlace routability: ~1.05–1.15 (estimated)
- After SA: ~0.97–1.06

This would put us at rank 2–8 depending on actual congestion reduction.
**Uncertainty is high** — the real number depends on DREAMPlace's ability to move
macros to lower-congestion regions without degrading WL/density.

## Does This Solve the Local Minima Problem?

**Partially.** 

DREAMPlace starting from Xplace positions is still starting from Xplace's
WL+Density basin. The congestion penalty will push macros toward less-congested areas,
but the WL+Density gradients resist this. The result is a compromise local minimum —
better than Xplace alone, but not as good as if congestion had been in the objective
from the start.

The 100-seed approach **indirectly helps**: different noise ratios (0.005–0.20) cause
Xplace to converge to different basins. Some of those basins happen to be naturally
less congested. DREAMPlace routability from the less-congested basins can reach a
better final minimum. Taking the best of K=3 DREAMPlace runs exploits this diversity.

The true fix for the local minima problem would be a custom differentiable GP that
includes congestion from the start (Option 2 in HANDOFF.md). This design is a
practical stepping stone.

## Risks

| Risk | Mitigation |
|------|-----------|
| Entropy injection still fires | Verify overflow < 0.95 at DREAMPlace start; add `iteration > 100` guard |
| DREAMPlace CUDA 12.4 incompatibility | Apply all 5 patches from patch_dreamplace.sh; test build |
| DREAMPlace degrades WL | Monitor wl_cost separately; if routability_weight too high, reduce |
| Docker rebuild fails (WSL2 crash) | Use `docker exec` build approach (HANDOFF.md Docker section) |
| DREAMPlace too slow per run | Reduce inner_iter to 500; reduce K to 2 or 1 |
| DREAMPlace creates overlaps | Always run _legalize_hard after DREAMPlace output |

## What NOT To Do

- Do NOT use `--routability_opt_flag` with center initialization (entropy injection)
- Do NOT use DREAMPlace CT-init positions (creates 139-200 overlaps)
- Do NOT try Xplace route_force again (CUDA kernel buffer overflow, unfixable)
- Do NOT run DREAMPlace for full budget (it should converge fast from Xplace init)

## Alternative: If DREAMPlace Rebuild Fails

If the Docker build fails or DREAMPlace is too slow, fall back to the custom RUDY
approach but applied DURING Xplace initialization rather than post-GP:

1. Compute RUDY from CT positions using TILOS API
2. Use RUDY gradient to push macros to low-congestion regions
3. Use those pre-adjusted positions as Xplace starting points (via bookshelf .pl)
4. Run standard Xplace GP from those positions

This is cheaper than DREAMPlace (no Docker rebuild) but weaker (congestion only in
initialization, not during GP).
