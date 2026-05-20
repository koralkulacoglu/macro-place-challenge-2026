# Design: Custom Differentiable Global Placer with Congestion Awareness

## Problem

Xplace / ePlace finds a Wirelength + Density minimum but is blind to routing capacity, causing severe routing bottlenecks. We need a Global Placer (GP) that directly minimizes our exact target proxy cost scoring function:

$$\text{Proxy Cost} = 1.0 \times \text{Wirelength} + 0.5 \times \text{Density} + 0.5 \times \text{Congestion}$$

## Core Idea

Run gradient descent directly on macro positions using the proxy cost as the loss function. All three terms will be made fully differentiable in PyTorch using `torch.autograd`. We will use the Adam optimizer, leverage multiple random restarts to escape local minima, and execute everything natively in Python without external tools or Docker rebuilds.

## The Three Terms

### 1. Wirelength (HPWL)

HPWL is non-differentiable due to max/min operations. We approximate it using the standard log-sum-exp smooth approximation:

$$\text{HPWL}_{\text{smooth}}(net) = \alpha \ln \left( \sum e^{x_i/\alpha} \right) - \alpha \ln \left( \sum e^{-x_i/\alpha} \right) + \text{[same for } y\text{]}$$

- **Implementation:** Use `torch.logsumexp` on pin positions (one call per net per axis). Pin positions are computed as $\text{macro center} + \text{pin offset}$ via `benchmark.macro_pin_offsets`. For soft macros without offsets, the center is used directly. $\alpha$ acts as a temperature parameter controlling smoothness.

### 2. Density Penalty (With Global Spreading)

A naive bin-based overlap penalty only provides local gradients, meaning clumpy macros won't feel a pull toward empty spaces on the canvas.

- **Implementation:** Compute a grid of dimensions `grid_rows × grid_cols`. Each macro contributes to its overlapping bins using a smooth triangle or cosine kernel (~30 lines of PyTorch).
- **Refinement (Gaussian Blur):** To simulate global forces without a complex FFT Poisson solver, apply a 2D Gaussian blur via `torch.nn.functional.conv2d` to the generated density grid. Start with a large blur radius ($\sigma$) so macros feel long-range repulsive forces toward empty regions, and anneal $\sigma$ down over the optimization iterations.

### 3. Congestion (Differentiable RUDY)

Rectangular Uniform wire Density (RUDY) measures routing demand. Standard bounding box calculation using `torch.min` or `torch.max` yields zero gradients for all internal pins of a net.

- **Implementation:** Map nets to the gcell grid. Capacity limits are read directly from `benchmark.hroutes_per_micron` and `benchmark.vroutes_per_micron`.
- **Refinement (LSE Bounding Boxes):** Use the exact same log-sum-exp smooth approximation from the wirelength term to find the boundary coordinates ($x_{\min}, x_{\max}, y_{\min}, y_{\max}$) of each net. This ensures that every single pin connected to a congested net receives a non-zero gradient pulling it inward to collapse the routing footprint.

---

## Legality & Refinement

During gradient descent, macros will overlap freely. We resolve this through a combination of mathematical schedules and local search:

- **Dynamic Weight Scheduling:** Start the optimizer with the density penalty weight near `0.0`. This allows macros to rapidly slide past each other and establish an optimal topological arrangement based purely on wirelength and congestion. Every 20–50 iterations, multiply the density weight by a scalar (e.g., `1.05`) to gradually force the macros to spread apart legally.
- **Coordinate Descent (CD) Polish:** Analytical placers struggle to pack large blocks tightly against one another without tiny overlaps. Before handing the positions over to Simulated Annealing (SA), execute a fast Coordinate Descent pass. Move one macro at a time along a localized grid to snap them into optimal, strictly legal positions.
- **Final Legalization:** Always run `_legalize_hard` after optimization ends before passing the solution to SA.

---

## Optimizer & Setup

- **Variables:** Position tensor of shape `[num_macros, 2]` with `requires_grad=True`.
- **Algorithm:** Adam optimizer with learning rate warmup.
- **Constraints:** Clamp positions to canvas boundaries after every optimizer step. For fixed macros, zero out their gradients prior to the optimizer step.

---

## Multiple Seeds / Restarts

To defeat the local minima problem inherent to macro placement, run the optimizer in parallel across diverse starting configurations:

1.  **The Reference Seed:** Current CT positions.
2.  **The Seed Baseline:** Xplace GP generated positions (provides a strong wirelength floor).
3.  **Noised Seeds:** CT positions injected with Gaussian noise at varying scales.
4.  **Exploratory Seeds:** Pure random coordinates scattered within the canvas bounds.

Evaluate all finalized placements after legalization using the exact `compute_proxy_cost` oracle and select the best overall performer.

---

## Integration into Pipeline

    [ Diverse Starting Seeds ]
    (Xplace Baseline / Random / Noised CT / CT)
                       │
                       ▼
              [ Custom PyTorch GP ]
    - Optimizes: 1.0×WL + 0.5×Density + 0.5×Congestion
    - Dynamic density annealing + Gaussian blur spreading
    - Log-sum-exp differentiable RUDY bounding boxes
                       │
                       ▼
          [ Coordinate Descent Polish ]
    - Rapid, single-macro localized legal adjustments
                       │
                       ▼
                _legalize_hard
                       │
                       ▼
            [ Sequential SA (Polish) ]
    - Runs on remaining time budget (~2500s)

---

## Expected Results

| Stage                      | ibm01 proxy | Avg (17 benchmarks) |
| :------------------------- | :---------- | :------------------ |
| Xplace GP (current)        | 0.908       | ~1.25               |
| Custom GP + CD (Estimated) | 0.80–0.85   | ~1.00–1.08          |
| + SA Polish                | 0.74–0.79   | ~0.92–0.98          |

_Note: The highest verified score on the leaderboard sits at 1.0109. Achieving an average sub-1.00 score secures a top tier position._

---

## Risks & Mitigations

| Risk                                   | Mitigation                                                                     |
| :------------------------------------- | :----------------------------------------------------------------------------- |
| Density term too weak / macros clump   | Increase final density weight ceiling; tune target density parameters.         |
| Zero-gradients stall internal net pins | Verified fix: Log-sum-exp smoothing replaces raw min/max box boundaries.       |
| macros get stuck in local clumps       | Verified fix: Apply 2D Gaussian blur conv2d to density grid for global forces. |
| Slow optimization per seed             | Limit Adam loop to 2000 iterations; batch independent seeds together.          |

## What NOT To Do

- Do **NOT** implement an FFT Poisson solver—2D convolution blur on the grid achieves the required macro spreading with zero architectural overhead.
- Do **NOT** use `torch.min` or `torch.max` for RUDY coordinate mapping.
- Do **NOT** use the baseline `compute_proxy_cost` inside the gradient loop (it contains non-differentiable C++ and Python hooks). Only use it as an evaluation oracle to select the winning seed.
