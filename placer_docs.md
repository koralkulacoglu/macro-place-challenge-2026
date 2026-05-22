# LKPlacer technical specification: The "Digital Twin" Blueprint

`LKPlacer` is a high-performance macro placer that exploits massive data parallelism on the GPU and optimized sequential search on the CPU. This document provides the low-level mathematical and architectural details required to recreate the system.

## 1. Phase α: Massive Parallel Global Placement (GPU)

The global placement phase (`gp.py`) is designed to explore the layout landscape using a **Parallel Tempering (Replica Exchange)** strategy.

### 1.1 Parallel Tempering Architecture
*   **Replica Batching:** The placer maintains a population of $K$ independent placement chains (replicas) stored as a single `[K, N, 2]` tensor.
*   **Jitter Ladder:** Each replica starts with a different level of random jitter (0% to 10% of canvas dimensions), ensuring diverse exploration of the solution space.
*   **Temperature Ladder:** Replicas are assigned "temperatures" on a geometric ladder. Hotter chains accept worse moves more frequently, allowing them to jump over high-cost "ridges" in the landscape.
*   **Metropolis Swaps:** Every 50 steps, the placer attempts to swap the physical configurations of adjacent chains. A swap is accepted if:
    $$P(swap) = \min(1, \exp((\frac{1}{T_k} - \frac{1}{T_{k+1}}) \cdot (Cost_{k+1} - Cost_k)))$$
    This pushes high-quality solutions toward "cold" chains for refinement while sending "stuck" solutions to "hot" chains for reshuffling.

### 1.2 Mathematical Primitives
*   **Poisson Potential Field:** The solver calculates the global repulsion field $\phi$ by solving $\nabla^2 \phi = -\rho$ using **2D Fast Fourier Transforms (FFTs)**. This allows a macro at one end of the chip to "feel" a congestion hotspot at the other end.
*   **Log-Sum-Exp (LSE) Smooth HPWL:** To make Manhattan wirelength differentiable, it uses:
    $$WL_{smooth} = \gamma \sum_{net} \log(\sum_{pin} \exp(x_i/\gamma)) + \gamma \sum_{net} \log(\sum_{pin} \exp(-x_i/\gamma))$$
    As $\gamma \to 0$, this converges to the true non-differentiable HPWL.
*   **Bilinear Density Spreading:** Macro area is spread onto the grid using a differentiable bilinear kernel, ensuring smooth gradients as macros cross grid boundaries.

## 2. Phase 1: The Optimized FastEvaluator (CPU)

The `FastEvaluator` is a "Digital Twin" of the competition's scoring engine. It is engineered for extreme throughput during the discrete refinement phases.

### 2.1 Corner Trick Vectorization (Prefix Sums)
To compute the **RUDY Congestion Map** in $O(N + R \times C)$ instead of $O(N \times R \times C)$, it uses a 2D prefix-sum optimization:
1.  For each net, it adds the demand value to the four corners of its bounding box in a "difference grid" $\Delta$.
2.  It performs a 2D cumulative sum (integration) across $\Delta$ to reconstruct the demand grid.
3.  This replaces millions of grid-slice updates with just $4N$ additions and one pass over the grid.

### 2.2 Incremental State Maintenance
When a single macro is moved, the evaluator does **not** recompute the whole chip. It:
1.  Subtracts the macro's area from the current `density_grid`.
2.  Subtracts the demand of all nets connected to that macro from the `congestion_grid`.
3.  Updates the macro's coordinates.
4.  Adds the new area and new net demands back into the grids.
This allows for $\sim 5,000+$ move evaluations per second on a single CPU core.

### 2.3 Bit-Exact Mirroring logic
*   **Star-L Topology:** High-fanout nets (4+ pins) are routed as a star centered at Pin 0, using Horizontal-first L-routes to every sink.
*   **1D Box Smoothing:** Vertical congestion is smoothed horizontally using a 1D window filter of size $2 \times \text{smooth\_range} + 1$; Horizontal congestion is smoothed vertically.
*   **ABU Aggregation:** The final congestion score is the **mean of the top 5%** of all combined H and V grid cells.

## 3. Phase 2 & 3: Discrete Combinatorial Refinement

### 3.1 Lin-Kernighan (LK) k-opt
*   **Multi-Step Swap Chains:** Instead of simple $A \leftrightarrow B$ swaps, the LK phase explores $A \to B \to C \to A$ cycles. 
*   **Gain-Based Search:** It only continues a chain if the intermediate partial moves show a "gain" potential, effectively pruning the massive $N!$ search space.

### 3.2 Late Acceptance Hill Climbing (LAHC)
*   **History Buffer:** LAHC maintains a list of the last 100 accepted costs. A new configuration is accepted if it is better than the *oldest* value in the buffer.
*   **Why LAHC:** Unlike Simulated Annealing, LAHC has no temperature parameters to tune. It automatically adapts its "greediness" based on the progress made in the last 100 steps, making it extremely robust for final polishing.

### 3.3 Biased Proposals
*   **Partner-Centroid Bias:** Moves soft macros toward the weighted geometric center of their neighbors:
    $$\vec{X}_{target} = \frac{\sum w_i \vec{X}_i}{\sum w_i}$$
*   **Decongestion Push:** Identifies macros sitting in "hot" cells and proposes moves in the direction of the negative congestion gradient: $\vec{V}_{move} \propto -\nabla Congestion$.

## 4. Operational Flow (Recap for Re-implementation)
1.  **Initialize $K$ replicas** on GPU; run 500 steps of Poisson GP with Parallel Tempering.
2.  **Transfer best replica to CPU**; Legalize via greedy push-apart + spiral.
3.  **Construct FastEvaluator**; Calibrate linear scale against the official oracle.
4.  **Execute LK Swap Chains** to resolve floorplan-level gridlocks.
5.  **Final LAHC Polish** with biased proposals to squeeze the last 1-2% out of the proxy score.
6.  **Validate final result** against bit-perfect oracle.
