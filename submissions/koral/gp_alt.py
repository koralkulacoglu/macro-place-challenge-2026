"""
Phase α — Focused Electrostatic Global Placement (Global Top-K Version)
=====================================================================

Alternative version of gp.py that uses an Evolutionary Selection strategy.
For each step, it evaluates J candidates (K replicas * 10 samples) and 
selects the absolute top K configurations to survive to the next iteration.
These survivors are assigned to the temperature ladder sorted by cost.
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


def _build_pin_tensors(benchmark: Benchmark, device: torch.device):
    n_hard = benchmark.num_hard_macros
    pin_offsets = benchmark.macro_pin_offsets
    n_nets = int(benchmark.num_nets)
    owners_list, offs_list, net_id_list, driver_idx_list = [], [], [], []
    total_pin_count = 0
    npn = benchmark.net_pin_nodes
    if not npn:
        for net_idx in range(n_nets):
            nodes = benchmark.net_nodes[net_idx].numpy().astype(np.int64)
            driver_abs_idx = total_pin_count
            for o in nodes:
                owners_list.append(o); offs_list.append([0.0, 0.0]); net_id_list.append(net_idx); driver_idx_list.append(driver_abs_idx); total_pin_count += 1
    else:
        for net_idx in range(n_nets):
            pn = npn[net_idx].numpy().astype(np.int64)
            if pn.size == 0: continue
            driver_abs_idx = total_pin_count
            for k in range(pn.shape[0]):
                o, s = int(pn[k, 0]), int(pn[k, 1])
                off_x, off_y = 0.0, 0.0
                if o < n_hard and pin_offsets and o < len(pin_offsets):
                    po = pin_offsets[o]
                    if po is not None and po.shape[0] > s: off_x, off_y = float(po[s, 0]), float(po[s, 1])
                owners_list.append(o); offs_list.append([off_x, off_y]); net_id_list.append(net_idx); driver_idx_list.append(driver_abs_idx); total_pin_count += 1
    return (
        torch.tensor(owners_list, device=device, dtype=torch.long),
        torch.tensor(offs_list, device=device, dtype=torch.float32),
        torch.tensor(net_id_list, device=device, dtype=torch.long),
        n_nets,
        torch.tensor(driver_idx_list, device=device, dtype=torch.long),
    )


def _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros):
    K = pop.shape[0]
    n_ports = port_pos.shape[0]
    owner_pos = torch.cat([pop, port_pos.unsqueeze(0).expand(K, n_ports, 2)], dim=1) if n_ports > 0 else pop
    return owner_pos[:, owner_idx, :] + pin_off.unsqueeze(0)


def smooth_hpwl(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, gamma, cw, ch, n_nets_norm, weights_per_net=None):
    K = pop.shape[0]
    pins = _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros)
    x, y = pins[..., 0], pins[..., 1]
    idx = net_id.unsqueeze(0).expand(K, -1)
    x_max = torch.full((K, n_nets), -1e9, device=pop.device, dtype=pop.dtype).scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min = torch.full((K, n_nets), 1e9, device=pop.device, dtype=pop.dtype).scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max = torch.full((K, n_nets), -1e9, device=pop.device, dtype=pop.dtype).scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min = torch.full((K, n_nets), 1e9, device=pop.device, dtype=pop.dtype).scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    lse_max_x = gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((x - x_max.gather(1, idx)) / gamma)).clamp(min=1e-12)) + x_max
    lse_min_x = -gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((x_min.gather(1, idx) - x) / gamma)).clamp(min=1e-12)) + x_min
    lse_max_y = gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((y - y_max.gather(1, idx)) / gamma)).clamp(min=1e-12)) + y_max
    lse_min_y = -gamma * torch.log(torch.zeros(K, n_nets, device=pop.device, dtype=pop.dtype).scatter_add_(1, idx, torch.exp((y_min.gather(1, idx) - y) / gamma)).clamp(min=1e-12)) + y_min
    hpwl = (lse_max_x - lse_min_x) + (lse_max_y - lse_min_y)
    if weights_per_net is not None: hpwl = hpwl * weights_per_net.unsqueeze(0)
    return hpwl.sum(dim=1) / ((cw + ch) * max(n_nets_norm, 1))


def bilinear_density(pop, sizes, grid_col, grid_row, cw, ch):
    K, device, dtype = pop.shape[0], pop.device, pop.dtype
    grid_area = (cw / grid_col) * (ch / grid_row)
    col_l = torch.arange(grid_col, device=device, dtype=dtype) * (cw / grid_col)
    row_b = torch.arange(grid_row, device=device, dtype=dtype) * (ch / grid_row)
    hw, hh = sizes[:, 0] / 2, sizes[:, 1] / 2
    x_ov = (torch.min((pop[..., 0] + hw).unsqueeze(-1), (col_l + (cw / grid_col)).view(1, 1, grid_col)) - torch.max((pop[..., 0] - hw).unsqueeze(-1), col_l.view(1, 1, grid_col))).clamp(min=0.0)
    y_ov = (torch.min((pop[..., 1] + hh).unsqueeze(-1), (row_b + (ch / grid_row)).view(1, 1, grid_row)) - torch.max((pop[..., 1] - hh).unsqueeze(-1), row_b.view(1, 1, grid_row))).clamp(min=0.0)
    return torch.einsum("knx,kny->kyx", x_ov, y_ov) / grid_area


def _smooth_1d_along_axis(grid, smooth_range, axis):
    sr, K = smooth_range, grid.shape[0]
    if axis == 0:
        R, C = grid.shape[-2:]
        cnt = (torch.clamp(torch.arange(C, device=grid.device) + sr, max=C - 1) - torch.clamp(torch.arange(C, device=grid.device) - sr, min=0) + 1).to(grid.dtype)
        pad = torch.nn.functional.pad(grid / cnt.view(1, 1, -1), (sr, sr, 0, 0))
        csum = torch.cumsum(pad, dim=-1)
        return csum[..., 2 * sr:] - torch.cat([torch.zeros(K, R, 1, device=grid.device, dtype=grid.dtype), csum[..., :C - 1 + 2 * sr]], dim=-1)[..., :C]
    else:
        R, C = grid.shape[-2:]
        cnt = (torch.clamp(torch.arange(R, device=grid.device) + sr, max=R - 1) - torch.clamp(torch.arange(R, device=grid.device) - sr, min=0) + 1).to(grid.dtype)
        pad = torch.nn.functional.pad(grid / cnt.view(1, -1, 1), (0, 0, sr, sr))
        csum = torch.cumsum(pad, dim=-2)
        return csum[..., 2 * sr:, :] - torch.cat([torch.zeros(K, 1, C, device=grid.device, dtype=grid.dtype), csum[..., :R - 1 + 2 * sr, :]], dim=-2)[..., :R, :]


def _poisson_solve_batched(rho):
    K, R, C = rho.shape
    rho_hat = torch.fft.fft2(rho)
    ky = torch.fft.fftfreq(R, d=1.0, device=rho.device, dtype=torch.float32) * 2 * math.pi
    kx = torch.fft.fftfreq(C, d=1.0, device=rho.device, dtype=torch.float32) * 2 * math.pi
    k2 = (ky.view(1, R, 1) ** 2 + kx.view(1, 1, C) ** 2).clamp(min=1e-9)
    phi_hat = rho_hat / k2
    phi_hat[:, 0, 0] = 0.0
    return torch.fft.ifft2(phi_hat).real


def focused_electrostatic_loss(density_grid, top_k_frac, push):
    K, R, C = density_grid.shape
    flat = density_grid.reshape(K, -1)
    threshold = torch.sort(flat, dim=1, descending=True)[0][:, max(1, int(R * C * top_k_frac)) - 1:max(1, int(R * C * top_k_frac))].view(K, 1, 1)
    source = torch.clamp(density_grid - threshold, min=0.0) * push
    source = source - source.mean(dim=(1, 2), keepdim=True)
    return (_poisson_solve_batched(source) ** 2).mean(dim=(1, 2))


def rudy_demand(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, grid_col, grid_row, cw, ch, h_per_um, v_per_um):
    K, device, dtype = pop.shape[0], pop.device, pop.dtype
    pins = _pin_positions(pop, owner_idx, pin_off, port_pos, n_macros)
    x, y, idx = pins[..., 0], pins[..., 1], net_id.unsqueeze(0).expand(K, -1)
    x_max = torch.full((K, n_nets), -1e9, device=device, dtype=dtype).scatter_reduce_(1, idx, x, reduce="amax", include_self=True)
    x_min = torch.full((K, n_nets), 1e9, device=device, dtype=dtype).scatter_reduce_(1, idx, x, reduce="amin", include_self=True)
    y_max = torch.full((K, n_nets), -1e9, device=device, dtype=dtype).scatter_reduce_(1, idx, y, reduce="amax", include_self=True)
    y_min = torch.full((K, n_nets), 1e9, device=device, dtype=dtype).scatter_reduce_(1, idx, y, reduce="amin", include_self=True)
    bbox_w, bbox_h = (x_max - x_min).clamp(min=1e-3), (y_max - y_min).clamp(min=1e-3)
    grid_w, grid_h = cw / grid_col, ch / grid_row
    col_l = torch.arange(grid_col, device=device, dtype=dtype) * grid_w
    row_b = torch.arange(grid_row, device=device, dtype=dtype) * grid_h
    x_ov = (torch.min(x_max.unsqueeze(-1), (col_l + grid_w).view(1, 1, grid_col)) - torch.max(x_min.unsqueeze(-1), col_l.view(1, 1, grid_col))).clamp(min=0.0)
    y_ov = (torch.min(y_max.unsqueeze(-1), (row_b + grid_h).view(1, 1, grid_row)) - torch.max(y_min.unsqueeze(-1), row_b.view(1, 1, grid_row))).clamp(min=0.0)
    h_demand = torch.einsum("knx,kny->kyx", x_ov, y_ov * (1.0 / bbox_h).unsqueeze(-1).expand_as(y_ov))
    v_demand = torch.einsum("knx,kny->kyx", x_ov * (1.0 / bbox_w).unsqueeze(-1).expand_as(x_ov), y_ov)
    return v_demand / (grid_w * v_per_um), h_demand / (grid_h * h_per_um)


def focused_congestion_loss(v_demand, h_demand, smooth_range, top_k_frac=0.05):
    K = v_demand.shape[0]
    v_s, h_s = _smooth_1d_along_axis(v_demand, smooth_range, axis=0), _smooth_1d_along_axis(h_demand, smooth_range, axis=1)
    combined = torch.cat([v_s.reshape(K, -1), h_s.reshape(K, -1)], dim=1)
    k = max(1, int(combined.shape[1] * top_k_frac))
    return torch.sort(combined, dim=1, descending=True)[0][:, :k].mean(dim=1)


def pairwise_overlap(pos_hard, sizes_hard):
    K, N, _ = pos_hard.shape
    if N <= 1: return torch.zeros(K, device=pos_hard.device)
    xi, xj, yi, yj = pos_hard[:, :, 0:1], pos_hard[:, None, :, 0], pos_hard[:, :, 1:2], pos_hard[:, None, :, 1]
    wi, wj, hi, hj = sizes_hard[:, 0:1], sizes_hard[None, :, 0], sizes_hard[:, 1:2], sizes_hard[None, :, 1]
    ox, oy = ((wi + wj) / 2 - (xi - xj).abs()).clamp(min=0.0), ((hi + hj) / 2 - (yi - yj).abs()).clamp(min=0.0)
    mask = 1 - torch.eye(N, device=pos_hard.device).view(1, N, N)
    return ((ox.squeeze(-1) * oy.squeeze(-1)) * mask).sum(dim=(1, 2)) / 2


def run_global_placement(benchmark, plc=None, *, pop_size=4, n_steps=500, lr=0.03, gamma_start=1.0, gamma_end=0.05, density_w_start=0.0, density_w_end=1.0, cong_w_start=0.0, cong_w_end=1.0, overlap_w_start=0.0, overlap_w_end=200.0, top_k_density=0.10, top_k_cong=0.05, push_factor=1.0, smooth_range=2, time_budget_s=120.0, seed=0, replica_swap_every=50, replica_temperatures=None, verbose=True, log_every=50, use_dfg=False):
    from macro_place.objective import compute_proxy_cost
    from .placer import FastEvaluator
    device = _device()
    torch.manual_seed(seed); np.random.seed(seed)
    n_hard, n_macros = benchmark.num_hard_macros, benchmark.num_macros
    cw, ch = float(benchmark.canvas_width), float(benchmark.canvas_height)
    grid_col, grid_row = int(benchmark.grid_cols), int(benchmark.grid_rows)
    h_per_um, v_per_um = float(benchmark.hroutes_per_micron), float(benchmark.vroutes_per_micron)
    sizes = benchmark.macro_sizes.to(device).float()
    sizes_hard, half_w, half_h = sizes[:n_hard], sizes[:, 0] / 2, sizes[:, 1] / 2
    port_pos = benchmark.port_positions.to(device).float() if benchmark.port_positions.shape[0] > 0 else torch.zeros(0, 2, device=device)
    tensors = _build_pin_tensors(benchmark, device)
    owner_idx, pin_off, net_id, n_nets = tensors[0], tensors[1], tensors[2], tensors[3]
    
    # Stochastic J-Sampling: J = 10 * K
    S_PER_R = 10 
    TOTAL_S = pop_size * S_PER_R
    
    v_max = ((0.2 * cw) / 1.645) ** 2
    v_min = 1e-6
    if replica_temperatures is None:
        replica_temperatures = tuple(v_min * ((v_max / v_min) ** (k / (pop_size - 1))) for k in range(pop_size))
    
    weights_per_net, n_nets_norm = None, n_nets
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

    pop = benchmark.macro_positions.to(device).float().unsqueeze(0).expand(pop_size, -1, -1).contiguous().clone().requires_grad_(True)
    opt = torch.optim.Adam([pop], lr=lr)
    fixed_mask, fixed_pos = benchmark.macro_fixed.to(device), benchmark.macro_positions.to(device).float()
    
    dyn_net_weights = weights_per_net.clone() if weights_per_net is not None else torch.ones(n_nets, device=device)
    reweight_every, reweight_alpha = 60, 3.0
    
    fe = FastEvaluator(benchmark, plc) if plc is not None else None
    t0, history = time.time(), {"wl": [], "dens": [], "cong": [], "ov": [], "total": [], "real_proxy": []}
    sigma_t = torch.sqrt(torch.tensor(replica_temperatures, device=device, dtype=torch.float32).view(-1, 1, 1))

    for step in range(n_steps):
        if time.time() - t0 > time_budget_s: break
        progress = step / max(n_steps - 1, 1)
        gamma = gamma_start * (gamma_end / gamma_start) ** progress
        density_w, cong_w, overlap_w = density_w_start + (density_w_end - density_w_start) * progress, cong_w_start + (cong_w_end - cong_w_start) * progress, overlap_w_start * (1 - progress) + overlap_w_end * progress
        
        # 1. Gradient Step
        opt.zero_grad()
        wl = smooth_hpwl(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, gamma, cw, ch, n_nets_norm, weights_per_net=dyn_net_weights)
        dens_grid = bilinear_density(pop, sizes, grid_col, grid_row, cw, ch)
        dens_loss = focused_electrostatic_loss(dens_grid, top_k_density, push_factor)
        v_dem, h_dem = rudy_demand(pop, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, grid_col, grid_row, cw, ch, h_per_um, v_per_um)
        cong_loss = focused_congestion_loss(v_dem, h_dem, smooth_range, top_k_cong)
        ov_loss = pairwise_overlap(pop[:, :n_hard], sizes_hard) / (cw * ch)
        total_loss = (wl + density_w * dens_loss + cong_w * cong_loss + overlap_w * ov_loss)
        total_loss.sum().backward()
        with torch.no_grad(): pop.grad[:, fixed_mask] = 0
        opt.step() 
        
        # 2. Stochastic J-Sampling Lookahead (GLOBAL TOP-K SELECTION)
        with torch.no_grad():
            grad_pos = pop.unsqueeze(1).expand(-1, S_PER_R, -1, -1).clone()
            noise = torch.randn_like(grad_pos[:, 1:]) * sigma_t.unsqueeze(1)
            grad_pos[:, 1:] += noise
            
            cand = grad_pos.reshape(TOTAL_S, n_macros, 2)
            cand[:, fixed_mask] = fixed_pos[fixed_mask].unsqueeze(0).expand(TOTAL_S, -1, -1)
            cand[..., 0].clamp_(min=half_w, max=cw - half_w); cand[..., 1].clamp_(min=half_h, max=ch - half_h)
            
            # Batch Evaluate Pool
            c_wl = smooth_hpwl(cand, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, gamma, cw, ch, n_nets_norm, weights_per_net=dyn_net_weights)
            c_dens = focused_electrostatic_loss(bilinear_density(cand, sizes, grid_col, grid_row, cw, ch), top_k_density, push_factor)
            v_d, h_d = rudy_demand(cand, owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, grid_col, grid_row, cw, ch, h_per_um, v_per_um)
            c_cong = focused_congestion_loss(v_d, h_d, smooth_range, top_k_cong)
            c_ov = pairwise_overlap(cand[:, :n_hard], sizes_hard) / (cw * ch)
            c_total = c_wl + density_w * c_dens + cong_w * c_cong + overlap_w * c_ov
            
            # SURVIVAL OF THE FITTEST: Top-K out of all J candidates
            best_vals, best_global_idx = torch.topk(c_total, k=pop_size, largest=False)
            best_pos_batch = cand[best_global_idx]
            
            # Reassign to pop, sorted such that best is at index 0 (coldest)
            pop.data[...] = best_pos_batch

            # Record history of the POPULATION MEAN
            history["wl"].append(c_wl.mean().item())
            history["dens"].append(c_dens.mean().item())
            history["cong"].append(c_cong.mean().item())
            history["ov"].append(c_ov.mean().item())
            history["total"].append(c_total.mean().item())

            if fe is not None:
                # Still track real_proxy for comparison, using mean across population
                k_proxy = []
                for k in range(pop_size):
                    fe.positions = pop[k].detach().cpu().numpy().astype(np.float64); fe._init_caches(); c = fe.proxy_cost()
                    k_proxy.append(c["proxy_cost"])
                history["real_proxy"].append(sum(k_proxy) / len(k_proxy))

        if step > 0 and step % reweight_every == 0:
            with torch.no_grad():
                v_d, h_d = rudy_demand(pop[0:1], owner_idx, pin_off, net_id, n_nets, port_pos, n_macros, grid_col, grid_row, cw, ch, h_per_um, v_per_um)
                v_s, h_s = _smooth_1d_along_axis(v_d, smooth_range, axis=0), _smooth_1d_along_axis(h_d, smooth_range, axis=1)
                comb = (v_s + h_s).reshape(-1); thresh = torch.sort(comb, descending=True)[0][max(1, int(comb.shape[0]*0.05))-1]
                pins = _pin_positions(pop[0:1], owner_idx, pin_off, port_pos, n_macros)
                x, y, idx = pins[..., 0], pins[..., 1], net_id.unsqueeze(0).expand(1, -1)
                xn, xx = torch.full((1, n_nets), 1e9, device=device).scatter_reduce_(1, idx, x, reduce="amin"), torch.full((1, n_nets), -1e9, device=device).scatter_reduce_(1, idx, x, reduce="amax")
                yn, yx = torch.full((1, n_nets), 1e9, device=device).scatter_reduce_(1, idx, y, reduce="amin"), torch.full((1, n_nets), -1e9, device=device).scatter_reduce_(1, idx, y, reduce="amax")
                c0, c1, r0, r1 = (xn[0]/cw*grid_col).long().clamp(0, grid_col-1), (xx[0]/cw*grid_col).long().clamp(0, grid_col-1), (yn[0]/ch*grid_row).long().clamp(0, grid_row-1), (yx[0]/ch*grid_row).long().clamp(0, grid_row-1)
                hot_mask = torch.zeros(n_nets, device=device, dtype=torch.bool); grid_hot = (v_s[0] + h_s[0]) > thresh
                for n in range(n_nets):
                    if grid_hot[r0[n]:r1[n]+1, c0[n]:c1[n]+1].any(): hot_mask[n] = True
                dyn_net_weights = torch.where(hot_mask, weights_per_net * reweight_alpha, weights_per_net) if weights_per_net is not None else torch.where(hot_mask, torch.ones(n_nets, device=device)*reweight_alpha, torch.ones(n_nets, device=device))

        if verbose and step % log_every == 0:
            print(f"  [GP-Alt] step {step:4d} wl={history['wl'][-1]:.4f} dens={history['dens'][-1]:.4f} cong={history['cong'][-1]:.4f} ov={history['ov'][-1]:.5f}", flush=True)
    
    best_cost, best_pos = float("inf"), None
    for k in range(pop_size):
        pos_np = pop[k].detach().cpu().numpy().astype(np.float64)
        if plc is not None:
            c = compute_proxy_cost(torch.from_numpy(pos_np).float(), benchmark, plc)
            score = float(c["proxy_cost"]) + (10.0 if c["overlap_count"] > 0 else 0.0)
            if score < best_cost: best_cost, best_pos = score, pos_np
        else: best_pos = pos_np; break
    return best_pos, history
