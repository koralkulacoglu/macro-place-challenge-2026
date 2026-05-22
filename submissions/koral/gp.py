"""
Phase α — Focused Electrostatic Global Placement
================================================

Continuous global placer that runs **before** the LK + LAHC refinement
pipeline.  Three innovations vs classical ePlace / DREAMPlace:

1. **Focused Poisson density target**.  Standard ePlace pushes every grid
   cell toward uniform density.  Our proxy only penalises the **top 10%**
   of density cells (and top 5% of routing cells).  So we set the Poisson
   source term = `density - top10pct_threshold` clamped at zero — only
   hot cells generate a field.  This concentrates electrostatic force where
   it lowers the actual cost.

2. **Focused congestion gradient**.  Same idea applied to RUDY routing
   demand: only cells above the top-5% threshold contribute to the loss.

3. **Pure 2D FFT Poisson solver** for the density potential.  Replaces the
   local bilinear-density gradient (which only feels its 4 neighbours)
   with a global potential field.  A density hotspot at (5, 5) creates a
   long-range field that pushes macros from (5, 5) toward emptier regions
   across the whole canvas.

The output is a numpy positions array `[N, 2]` that the downstream Phase 0
legalizer + LK + LAHC pipeline polishes.

Usage
-----
    from submissions.koral.gp import run_global_placement
    positions = run_global_placement(benchmark, plc, time_budget_s=120.0)
"""
from __future__ import annotations

import math
import time
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from macro_place.benchmark import Benchmark


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ────────────────────────────────────────────────────────────────────────────
# Pin tensor builder
# ────────────────────────────────────────────────────────────────────────────


def _build_pin_tensors(benchmark: Benchmark, device: torch.device):
    """Flatten net_pin_nodes into 4 parallel 1-D tensors.

    Returns
    -------
    owner_idx : [total_pins] int64
    pin_off   : [total_pins, 2]
    net_id    : [total_pins] int64
    n_nets    : int
    driver_idx_per_pin : [total_pins] int64 (optional, used for Star-L routing)
    """
    n_hard = benchmark.num_hard_macros
    pin_offsets = benchmark.macro_pin_offsets
    n_nets = int(benchmark.num_nets)
    if not benchmark.net_pin_nodes:
        # Fallback: use net_nodes (center pins only)
        owners_list = []
        offs_list = []
        net_id_list = []
        driver_idx_list = []
        total_pin_count = 0
        for net_idx in range(n_nets):
            nodes = benchmark.net_nodes[net_idx].numpy().astype(np.int64)
            driver_abs_idx = total_pin_count
            for o in nodes:
                owners_list.append(o)
                offs_list.append([0.0, 0.0])
                net_id_list.append(net_idx)
                driver_idx_list.append(driver_abs_idx)
                total_pin_count += 1
        return (
            torch.tensor(owners_list, device=device, dtype=torch.long),
            torch.tensor(offs_list, device=device, dtype=torch.float32),
            torch.tensor(net_id_list, device=device, dtype=torch.long),
            n_nets,
            torch.tensor(driver_idx_list, device=device, dtype=torch.long),
        )
    owners_list = []
    offs_list = []
    net_id_list = []
    driver_idx_list = []
    total_pin_count = 0
    for net_idx in range(n_nets):
        pn = benchmark.net_pin_nodes[net_idx].numpy().astype(np.int64)
        n_pins = pn.shape[0]
        if n_pins == 0: continue
        driver_abs_idx = total_pin_count
        for k in range(n_pins):
            o, s = int(pn[k, 0]), int(pn[k, 1])
            off_x, off_y = 0.0, 0.0
            if o < n_hard and pin_offsets and o < len(pin_offsets):
                po = pin_offsets[o]
                if po is not None and po.shape[0] > s:
                    off_x, off_y = float(po[s, 0]), float(po[s, 1])
            owners_list.append(o)
            offs_list.append([off_x, off_y])
            net_id_list.append(net_idx)
            driver_idx_list.append(driver_abs_idx)
            total_pin_count += 1
    return (
        torch.tensor(owners_list, device=device, dtype=torch.long),
        torch.tensor(offs_list, device=device, dtype=torch.float32),
        torch.tensor(net_id_list, device=device, dtype=torch.long),
        n_nets,
        torch.tensor(driver_idx_list, device=device, dtype=torch.long),
    )


# ────────────────────────────────────────────────────────────────────────────
# Differentiable cost components
# ────────────────────────────────────────────────────────────────────────────


def _pin_positions(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,      # [P]
    pin_off: torch.Tensor,        # [P, 2]
    port_pos: torch.Tensor,       # [n_ports, 2]
    n_macros: int,
) -> torch.Tensor:                # returns [K, P, 2]
    """Build pin positions: owner_pos + pin_off for macros; port_pos for ports."""
    K = pop.shape[0]
    n_ports = port_pos.shape[0]
    if n_ports > 0:
        ports_b = port_pos.unsqueeze(0).expand(K, n_ports, 2)
        owner_pos = torch.cat([pop, ports_b], dim=1)
    else:
        owner_pos = pop
    return owner_pos[:, owner_idx, :] + pin_off.unsqueeze(0)


def smooth_hpwl(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_macros: int,
    gamma: float,
    cw: float, ch: float,
    n_nets_norm: int,
    weights_per_net: Optional[torch.Tensor] = None,
) -> torch.Tensor:                # [K]
    """Weighted-Average HPWL via log-sum-exp.  Matches plc.get_cost normalization
    (divide by (cw + ch) × n_nets_norm).  As gamma → 0 this → true HPWL.
    """
    K = pop.shape[0]
    pins = _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros)  # [K, P, 2]
    x = pins[..., 0]
    y = pins[..., 1]
    idx = net_id.unsqueeze(0).expand(K, -1)
    big = 1e9
    # Per-net true max / min (for shifted LSE — numerically stable).
    x_max = torch.full((K, n_nets), -big, device=pop.device, dtype=pop.dtype)
    x_min = torch.full((K, n_nets), big, device=pop.device, dtype=pop.dtype)
    y_max = torch.full((K, n_nets), -big, device=pop.device, dtype=pop.dtype)
    y_min = torch.full((K, n_nets), big, device=pop.device, dtype=pop.dtype)
    x_max.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min.scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max.scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min.scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    # LSE
    lse_max_x = gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((x - x_max.gather(1, idx)) / gamma)).clamp(min=1e-12)) + x_max
    lse_min_x = -gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((x_min.gather(1, idx) - x) / gamma)).clamp(min=1e-12)) + x_min
    lse_max_y = gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((y - y_max.gather(1, idx)) / gamma)).clamp(min=1e-12)) + y_max
    lse_min_y = -gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((y_min.gather(1, idx) - y) / gamma)).clamp(min=1e-12)) + y_min
    hpwl = (lse_max_x - lse_min_x) + (lse_max_y - lse_min_y)  # [K, n_nets]
    if weights_per_net is not None:
        hpwl = hpwl * weights_per_net.unsqueeze(0)
    total_hpwl = hpwl.sum(dim=1)
    return total_hpwl / ((cw + ch) * max(n_nets_norm, 1))


def bilinear_density(
    pop: torch.Tensor,
    sizes: torch.Tensor,
    grid_col: int,
    grid_row: int,
    cw: float,
    ch: float,
) -> torch.Tensor:
    """Bilinear area-spreading onto a [grid_row, grid_col] grid. [K, R, C] out."""
    K, N, _ = pop.shape
    grid_area = (cw / grid_col) * (ch / grid_row)
    col_l = torch.arange(grid_col, device=pop.device, dtype=pop.dtype) * (cw / grid_col)
    col_r = col_l + (cw / grid_col)
    row_b = torch.arange(grid_row, device=pop.device, dtype=pop.dtype) * (ch / grid_row)
    row_t = row_b + (ch / grid_row)
    x, y = pop[..., 0], pop[..., 1]
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    macro_l = (x - hw.unsqueeze(0)).unsqueeze(-1)
    macro_r = (x + hw.unsqueeze(0)).unsqueeze(-1)
    macro_b_ = (y - hh.unsqueeze(0)).unsqueeze(-1)
    macro_t_ = (y + hh.unsqueeze(0)).unsqueeze(-1)
    col_lb, col_rb = col_l.view(1, 1, grid_col), col_r.view(1, 1, grid_col)
    row_bb, row_tb = row_b.view(1, 1, grid_row), row_t.view(1, 1, grid_row)
    x_ov = (torch.min(macro_r, col_rb) - torch.max(macro_l, col_lb)).clamp(min=0.0)
    y_ov = (torch.min(macro_t_, row_tb) - torch.max(macro_b_, row_bb)).clamp(min=0.0)
    occupied = torch.einsum("knx,kny->kyx", x_ov, y_ov)  # [K, R, C]
    return occupied / grid_area


def _smooth_1d_along_axis(grid: torch.Tensor, smooth_range: int, axis: int) -> torch.Tensor:
    sr = smooth_range
    K = grid.shape[0]
    if axis == 0:
        R, C = grid.shape[-2:]
        cols = torch.arange(C, device=grid.device)
        lp = torch.clamp(cols - sr, min=0)
        rp = torch.clamp(cols + sr, max=C - 1)
        cnt = (rp - lp + 1).to(grid.dtype)
        scaled = grid / cnt.view(1, 1, -1)
        pad = torch.nn.functional.pad(scaled, (sr, sr, 0, 0))
        csum = torch.cumsum(pad, dim=-1)
        cs0 = csum[..., 2 * sr:]
        cs1 = torch.cat([torch.zeros(K, R, 1, device=grid.device, dtype=grid.dtype), csum[..., :C - 1 + 2 * sr]], dim=-1)
        cs1 = cs1[..., :C]
        return cs0[..., :C] - cs1
    else:
        R, C = grid.shape[-2:]
        rows = torch.arange(R, device=grid.device)
        lp = torch.clamp(rows - sr, min=0)
        up = torch.clamp(rows + sr, max=R - 1)
        cnt = (up - lp + 1).to(grid.dtype)
        scaled = grid / cnt.view(1, -1, 1)
        pad = torch.nn.functional.pad(scaled, (0, 0, sr, sr))
        csum = torch.cumsum(pad, dim=-2)
        cs0 = csum[..., 2 * sr:, :]
        cs1 = torch.cat([torch.zeros(K, 1, C, device=grid.device, dtype=grid.dtype), csum[..., :R - 1 + 2 * sr, :]], dim=-2)
        cs1 = cs1[..., :R, :]
        return cs0[..., :R, :] - cs1


def _poisson_solve_batched(rho: torch.Tensor) -> torch.Tensor:
    K, R, C = rho.shape
    rho_hat = torch.fft.fft2(rho)
    ky = torch.fft.fftfreq(R, d=1.0, device=rho.device, dtype=torch.float32) * 2 * math.pi
    kx = torch.fft.fftfreq(C, d=1.0, device=rho.device, dtype=torch.float32) * 2 * math.pi
    k2 = ky.view(1, R, 1) ** 2 + kx.view(1, 1, C) ** 2
    k2 = k2.clamp(min=1e-9)
    phi_hat = rho_hat / k2
    phi_hat[:, 0, 0] = 0.0
    phi = torch.fft.ifft2(phi_hat).real
    return phi


def focused_electrostatic_loss(
    density_grid: torch.Tensor,   # [K, R, C]
    top_k_frac: float,
    push: float,
) -> torch.Tensor:
    K, R, C = density_grid.shape
    flat = density_grid.reshape(K, -1)
    n = R * C
    k = max(1, int(n * top_k_frac))
    sorted_d, _ = torch.sort(flat, dim=1, descending=True)
    threshold = sorted_d[:, k - 1:k]  # [K, 1]
    source = torch.clamp(density_grid - threshold.view(K, 1, 1), min=0.0)
    if push != 1.0: source = source * push
    source = source - source.mean(dim=(1, 2), keepdim=True)
    phi = _poisson_solve_batched(source)
    return (phi ** 2).mean(dim=(1, 2))


def rudy_demand(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_macros: int,
    grid_col: int, grid_row: int,
    cw: float, ch: float,
    h_per_um: float, v_per_um: float,
):
    K = pop.shape[0]
    device = pop.device
    dtype = pop.dtype
    pins = _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros)
    x, y = pins[..., 0], pins[..., 1]
    idx = net_id.unsqueeze(0).expand(K, -1)
    big = 1e9
    x_max = torch.full((K, n_nets), -big, device=device, dtype=dtype)
    x_min = torch.full((K, n_nets), big, device=device, dtype=dtype)
    y_max = torch.full((K, n_nets), -big, device=device, dtype=dtype)
    y_min = torch.full((K, n_nets), big, device=device, dtype=dtype)
    x_max.scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min.scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max.scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min.scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    bbox_w, bbox_h = (x_max - x_min).clamp(min=1e-3), (y_max - y_min).clamp(min=1e-3)
    grid_w, grid_h = cw / grid_col, ch / grid_row
    col_l = torch.arange(grid_col, device=device, dtype=dtype) * grid_w
    col_r, row_b = col_l + grid_w, torch.arange(grid_row, device=device, dtype=dtype) * grid_h
    row_t = row_b + grid_h
    bxl, bxr = x_min.unsqueeze(-1), x_max.unsqueeze(-1)
    byb, byt = y_min.unsqueeze(-1), y_max.unsqueeze(-1)
    col_lb, col_rb = col_l.view(1, 1, grid_col), col_r.view(1, 1, grid_col)
    row_bb, row_tb = row_b.view(1, 1, grid_row), row_t.view(1, 1, grid_row)
    x_ov = (torch.min(bxr, col_rb) - torch.max(bxl, col_lb)).clamp(min=0.0)
    y_ov = (torch.min(byt, row_tb) - torch.max(byb, row_bb)).clamp(min=0.0)
    inv_h, inv_w = (1.0 / bbox_h).unsqueeze(-1), (1.0 / bbox_w).unsqueeze(-1)
    h_demand = torch.einsum("knx,kny->kyx", x_ov, y_ov * inv_h.expand_as(y_ov))
    v_demand = torch.einsum("knx,kny->kyx", x_ov * inv_w.expand_as(x_ov), y_ov)
    v_demand = v_demand / (grid_w * v_per_um)
    h_demand = h_demand / (grid_h * h_per_um)
    return v_demand, h_demand


def focused_congestion_loss(
    v_demand: torch.Tensor,
    h_demand: torch.Tensor,
    smooth_range: int,
    top_k_frac: float = 0.05,
) -> torch.Tensor:
    K = v_demand.shape[0]
    v_s = _smooth_1d_along_axis(v_demand, smooth_range, axis=0)
    h_s = _smooth_1d_along_axis(h_demand, smooth_range, axis=1)
    combined = torch.cat([v_s.reshape(K, -1), h_s.reshape(K, -1)], dim=1)
    n = combined.shape[1]
    k = max(1, int(n * top_k_frac))
    sorted_c, _ = torch.sort(combined, dim=1, descending=True)
    return sorted_c[:, :k].mean(dim=1)


def dfg_demand(
    pop: torch.Tensor,            # [K, N, 2]
    owner_idx: torch.Tensor,
    pin_off: torch.Tensor,
    net_id: torch.Tensor,
    n_nets: int,
    port_pos: torch.Tensor,
    n_macros: int,
    grid_col: int, grid_row: int,
    cw: float, ch: float,
    h_per_um: float, v_per_um: float,
    driver_idx_per_pin: torch.Tensor,
):
    K = pop.shape[0]
    device, dtype = pop.device, pop.dtype
    pins = _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros)
    drivers, sinks = pins[:, driver_idx_per_pin, :], pins
    h_x_min, h_x_max, h_y_fixed = torch.min(drivers[..., 0], sinks[..., 0]), torch.max(drivers[..., 0], sinks[..., 0]), drivers[..., 1]
    v_y_min, v_y_max, v_x_fixed = torch.min(drivers[..., 1], sinks[..., 1]), torch.max(drivers[..., 1], sinks[..., 1]), sinks[..., 0]
    grid_w, grid_h = cw / grid_col, ch / grid_row
    col_l = torch.arange(grid_col, device=device, dtype=dtype) * grid_w
    col_r, row_b = col_l + grid_w, torch.arange(grid_row, device=device, dtype=dtype) * grid_h
    row_t = row_b + grid_h
    h_x_ov = (torch.min(h_x_max.unsqueeze(-1), col_r.view(1, 1, grid_col)) - torch.max(h_x_min.unsqueeze(-1), col_l.view(1, 1, grid_col))).clamp(min=0.0)
    h_y_rel = (h_y_fixed / grid_h).clamp(0, grid_row - 1.001)
    h_r0 = h_y_rel.long()
    h_r1, h_w1 = h_r0 + 1, h_y_rel - h_r0.float()
    h_w0 = 1.0 - h_w1
    v_y_ov = (torch.min(v_y_max.unsqueeze(-1), row_t.view(1, 1, grid_row)) - torch.max(v_y_min.unsqueeze(-1), row_b.view(1, 1, grid_row))).clamp(min=0.0)
    v_x_rel = (v_x_fixed / grid_w).clamp(0, grid_col - 1.001)
    v_c0 = v_x_rel.long()
    v_c1, v_w1 = v_c0 + 1, v_x_rel - v_c0.float()
    v_w0 = 1.0 - v_w1
    h_demand = torch.zeros(K, grid_row, grid_col, device=device, dtype=dtype)
    for k in range(K):
        h_demand[k].index_add_(0, h_r0[k], h_x_ov[k] * h_w0[k].unsqueeze(-1))
        h_demand[k].index_add_(0, h_r1[k], h_x_ov[k] * h_w1[k].unsqueeze(-1))
    v_demand = torch.zeros(K, grid_row, grid_col, device=device, dtype=dtype)
    for k in range(K):
        v_demand[k].index_add_(1, v_c0[k], v_y_ov[k].transpose(0, 1) * v_w0[k].unsqueeze(0))
        v_demand[k].index_add_(1, v_c1[k], v_y_ov[k].transpose(0, 1) * v_w1[k].unsqueeze(0))
    v_demand = v_demand / max(grid_w * v_per_um, 1e-9)
    h_demand = h_demand / max(grid_h * h_per_um, 1e-9)
    return v_demand, h_demand


def focused_dual_electrostatic_cong_loss(
    v_demand: torch.Tensor, h_demand: torch.Tensor,
    v_macro_block: torch.Tensor, h_macro_block: torch.Tensor,
    smooth_range: int, top_k_frac: float = 0.05,
) -> torch.Tensor:
    v_s = _smooth_1d_along_axis(v_demand + v_macro_block, smooth_range, axis=0)
    h_s = _smooth_1d_along_axis(h_demand + h_macro_block, smooth_range, axis=1)
    K, R, C = v_s.shape
    h_flat = h_s.reshape(K, -1)
    h_thresh = torch.sort(h_flat, dim=1, descending=True)[0][:, max(1, int(R*C*top_k_frac))-1:max(1, int(R*C*top_k_frac))].view(K, 1, 1)
    h_source = torch.clamp(h_s - h_thresh, min=0.0)
    h_phi = _poisson_solve_batched(h_source - h_source.mean(dim=(1, 2), keepdim=True))
    v_flat = v_s.reshape(K, -1)
    v_thresh = torch.sort(v_flat, dim=1, descending=True)[0][:, max(1, int(R*C*top_k_frac))-1:max(1, int(R*C*top_k_frac))].view(K, 1, 1)
    v_source = torch.clamp(v_s - v_thresh, min=0.0)
    v_phi = _poisson_solve_batched(v_source - v_source.mean(dim=(1, 2), keepdim=True))
    return (h_phi**2).mean(dim=(1, 2)) + (v_phi**2).mean(dim=(1, 2))


def pairwise_overlap(pos_hard: torch.Tensor, sizes_hard: torch.Tensor) -> torch.Tensor:
    K, N, _ = pos_hard.shape
    if N <= 1: return torch.zeros(K, device=pos_hard.device)
    xi, xj, yi, yj = pos_hard[:, :, 0:1], pos_hard[:, None, :, 0], pos_hard[:, :, 1:2], pos_hard[:, None, :, 1]
    wi, wj, hi, hj = sizes_hard[:, 0:1], sizes_hard[None, :, 0], sizes_hard[:, 1:2], sizes_hard[None, :, 1]
    ox, oy = ((wi + wj) / 2 - (xi - xj).abs()).clamp(min=0.0), ((hi + hj) / 2 - (yi - yj).abs()).clamp(min=0.0)
    mask = 1 - torch.eye(N, device=pos_hard.device).view(1, N, N)
    return ((ox.squeeze(-1) * oy.squeeze(-1)) * mask).sum(dim=(1, 2)) / 2


def run_global_placement(
    benchmark: Benchmark, plc=None, *, pop_size=4, n_steps=500, lr=0.03, gamma_start=1.0, gamma_end=0.05, density_w_start=0.0, density_w_end=1.0, cong_w_start=0.0, cong_w_end=1.0, overlap_w_start=0.0, overlap_w_end=200.0, top_k_density=0.10, top_k_cong=0.05, push_factor=1.0, smooth_range=2, time_budget_s=120.0, seed=0, replica_swap_every=50, replica_temperatures=None, verbose=True, log_every=50, use_dfg=False
) -> np.ndarray:
    from macro_place.objective import compute_proxy_cost
    device = _device()
    torch.manual_seed(seed); np.random.seed(seed)
    n_hard, n_macros = benchmark.num_hard_macros, benchmark.num_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    grid_col, grid_row = int(benchmark.grid_cols), int(benchmark.grid_rows)
    h_per_um, v_per_um = float(benchmark.hroutes_per_micron), float(benchmark.vroutes_per_micron)
    grid_w, grid_h = cw / grid_col, ch / grid_row
    sizes = benchmark.macro_sizes.to(device).float()
    sizes_hard, half_w, half_h = sizes[:n_hard], sizes[:, 0] / 2, sizes[:, 1] / 2
    port_pos = benchmark.port_positions.to(device).float() if benchmark.port_positions.shape[0] > 0 else torch.zeros(0, 2, device=device)
    tensors = _build_pin_tensors(benchmark, device)
    owner_idx, pin_off, net_id, n_nets = tensors[0], tensors[1], tensors[2], tensors[3]
    driver_idx_per_pin = tensors[4] if len(tensors) > 4 else None
    weights_per_net, n_nets_norm, h_alloc, v_alloc = None, n_nets, 0.0, 0.0
    if plc is not None:
        weights = np.ones(n_nets, dtype=np.float32)
        try:
            driver_names = list(plc.nets.keys())
            for i, name in enumerate(driver_names[:n_nets]):
                pin_i = plc.mod_name_to_indices[name]
                weights[i] = float(plc.modules_w_pins[pin_i].get_weight())
            n_nets_norm = int(getattr(plc, "net_cnt", n_nets))
        except Exception: pass
        weights_per_net = torch.tensor(weights, device=device, dtype=torch.float32)
        try: h_alloc, v_alloc = plc.get_macro_routing_allocation()
        except Exception: h_alloc, v_alloc = getattr(plc, "hrouting_alloc", 0.0), getattr(plc, "vrouting_alloc", 0.0)
    init_pos = benchmark.macro_positions.to(device).float()
    pop = init_pos.unsqueeze(0).expand(pop_size, -1, -1).contiguous().clone().requires_grad_(True)
    opt = torch.optim.Adam([pop], lr=lr)
    fixed_mask, fixed_pos = benchmark.macro_fixed.to(device), init_pos.clone()
    if replica_temperatures is None: replica_temperatures = tuple(0.01 * (2.0 ** k) for k in range(pop_size))
    swap_rng = np.random.default_rng(seed + 7)
    dyn_net_weights = weights_per_net.clone() if weights_per_net is not None else torch.ones(n_nets, device=device)
    t0 = time.time()
    for step in range(n_steps):
        if time.time() - t0 > time_budget_s: break
        progress = step / max(n_steps - 1, 1)
        gamma = gamma_start * (gamma_end / gamma_start) ** progress
        density_w, cong_w, overlap_w = density_w_start + (density_w_end - density_w_start) * progress, cong_w_start + (cong_w_end - cong_w_start) * progress, overlap_w_start * (1 - progress) + overlap_w_end * progress
        opt.zero_grad()
        wl = smooth_hpwl(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, gamma, cw, ch, n_nets_norm, weights_per_net=dyn_net_weights)
        dens_grid = bilinear_density(pop, sizes, grid_col, grid_row, cw, ch)
        dens_loss = focused_electrostatic_loss(dens_grid, top_k_density, push_factor)
        if use_dfg and driver_idx_per_pin is not None:
            v_dem, h_dem = dfg_demand(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, grid_col, grid_row, cw, ch, h_per_um, v_per_um, driver_idx_per_pin)
            cong_loss = focused_dual_electrostatic_cong_loss(v_dem, h_dem, bilinear_density(pop[:, :n_hard], sizes[:n_hard], grid_col, grid_row, cw, ch) * (v_alloc / max(grid_w * v_per_um, 1e-9)), bilinear_density(pop[:, :n_hard], sizes[:n_hard], grid_col, grid_row, cw, ch) * (h_alloc / max(grid_h * h_per_um, 1e-9)), smooth_range, top_k_cong)
        else:
            v_dem, h_dem = rudy_demand(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, grid_col, grid_row, cw, ch, h_per_um, v_per_um)
            cong_loss = focused_congestion_loss(v_dem, h_dem, smooth_range, top_k_cong)
        ov_loss = pairwise_overlap(pop[:, :n_hard], sizes_hard) / (cw * ch)
        total_loss = (wl + density_w * dens_loss + cong_w * cong_loss + overlap_w * ov_loss)
        total_loss.sum().backward()
        with torch.no_grad(): pop.grad[:, fixed_mask] = 0
        opt.step()
        with torch.no_grad():
            pop[:, fixed_mask] = fixed_pos[fixed_mask].unsqueeze(0).expand(pop_size, -1, -1)
            for a in [0, 1]: pop[..., a].clamp_(min=half_w if a==0 else half_h, max=(cw if a==0 else ch) - (half_w if a==0 else half_h))
            if pop_size > 1 and replica_swap_every > 0 and (step + 1) % replica_swap_every == 0:
                per_replica = total_loss.detach()
                for k in range(pop_size - 1):
                    Tk, Tkp1 = replica_temperatures[k], replica_temperatures[k + 1]
                    delta = (1.0 / Tk - 1.0 / Tkp1) * (per_replica[k+1].item() - per_replica[k].item())
                    if delta >= 0 or swap_rng.random() < math.exp(delta):
                        pop.data[k], pop.data[k+1] = pop.data[k+1].clone(), pop.data[k].clone()
                        per_replica[k], per_replica[k+1] = per_replica[k+1].clone(), per_replica[k].clone()
        if verbose and step % log_every == 0: print(f"  [GP] step {step:4d} wl={wl.mean():.4f} dens={dens_loss.mean():.4f} cong={cong_loss.mean():.4f} ov={ov_loss.mean():.5f}", flush=True)
    best_cost, best_pos = float("inf"), None
    for k in range(pop_size):
        pos_np = pop[k].detach().cpu().numpy().astype(np.float64)
        if plc is not None:
            c = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)
            score = float(c["proxy_cost"]) + (10.0 if c["overlap_count"] > 0 else 0.0)
            if score < best_cost: best_cost, best_pos = score, pos_np
        else: best_pos = pos_np; break
    return best_pos
