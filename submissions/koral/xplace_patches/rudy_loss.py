"""
Differentiable RUDY (Routing Utilization Density Estimation) congestion loss for Xplace GP.

Computes a smooth, autograd-differentiable RUDY density map from current macro positions
using log-sum-exp bounding boxes. Bypasses gpugr (which crashes on IBM ICCAD04 benchmarks).

Injected into calculator.py after the WL gradient when args.rudy_weight > 0.
"""

import torch


def rudy_congestion_loss(conn_node_pos, data):
    """
    Differentiable RUDY congestion loss. Returns scalar for autograd.

    RUDY: for each net, routing demand ≈ HPWL (length units).
    Demand is scattered to grid cells at the net's center via bilinear interpolation.
    Loss = sum(density_map^2) — penalizes routing demand concentration.
    """
    device = conn_node_pos.device
    dtype  = conn_node_pos.dtype
    nbx, nby = data.num_bin_x, data.num_bin_y
    lx, hx, ly, hy = data.die_info.tolist()
    bin_w = (hx - lx) / nbx
    bin_h = (hy - ly) / nby

    # Absolute pin positions: [num_pins, 2]
    pin_pos    = conn_node_pos[data.pin_id2node_id] + data.pin_rel_cpos
    hp_pin_pos = pin_pos[data.hyperedge_list]   # [total_hp_pins, 2]
    hp_x = hp_pin_pos[:, 0]
    hp_y = hp_pin_pos[:, 1]

    # net_ids[i] = net index for each entry in hyperedge_list
    num_nets = int(data.hyperedge_list_end.shape[0])
    ends     = data.hyperedge_list_end
    starts   = torch.cat([torch.zeros(1, dtype=torch.long, device=device), ends[:-1]])
    lengths  = ends - starts
    net_ids  = torch.repeat_interleave(torch.arange(num_nets, device=device), lengths)

    # Smooth bounding box via log-sum-exp: max ≈ (1/γ) log Σ_j exp(γ x_j)
    canvas_scale = max(hx - lx, hy - ly)
    gamma = 10.0 / canvas_scale

    def lse_max(vals, ids, n):
        s = torch.zeros(n, device=device, dtype=dtype).scatter_add_(
            0, ids, torch.exp(gamma * vals))
        return s.clamp(min=1e-30).log() / gamma

    def lse_min(vals, ids, n):
        s = torch.zeros(n, device=device, dtype=dtype).scatter_add_(
            0, ids, torch.exp(-gamma * vals))
        return -(s.clamp(min=1e-30).log()) / gamma

    x_max = lse_max(hp_x, net_ids, num_nets)
    x_min = lse_min(hp_x, net_ids, num_nets)
    y_max = lse_max(hp_y, net_ids, num_nets)
    y_min = lse_min(hp_y, net_ids, num_nets)

    bbox_w = (x_max - x_min).clamp(min=bin_w)
    bbox_h = (y_max - y_min).clamp(min=bin_h)
    hpwl   = (bbox_w + bbox_h)   # routing demand per net [num_nets]

    # Apply net mask
    if data.net_mask is not None:
        hpwl = hpwl * data.net_mask.to(dtype=dtype)

    # Scatter demand to grid via bilinear distribution at net center
    cx  = (x_max + x_min) * 0.5
    cy  = (y_max + y_min) * 0.5
    bx  = ((cx - lx) / bin_w - 0.5).clamp(0.0, nbx - 1.001)
    by_ = ((cy - ly) / bin_h - 0.5).clamp(0.0, nby - 1.001)

    bx0 = bx.long();   bx1 = (bx0 + 1).clamp(max=nbx - 1)
    by0 = by_.long();  by1 = (by0 + 1).clamp(max=nby - 1)
    fx  = (bx  - bx0.to(dtype=dtype)).to(dtype=dtype)
    fy  = (by_ - by0.to(dtype=dtype)).to(dtype=dtype)

    density_flat = torch.zeros(nbx * nby, device=device, dtype=dtype)
    density_flat.scatter_add_(0, bx0 * nby + by0, hpwl * (1.0 - fx) * (1.0 - fy))
    density_flat.scatter_add_(0, bx0 * nby + by1, hpwl * (1.0 - fx) * fy)
    density_flat.scatter_add_(0, bx1 * nby + by0, hpwl * fx * (1.0 - fy))
    density_flat.scatter_add_(0, bx1 * nby + by1, hpwl * fx * fy)

    density_map = density_flat.view(nbx, nby)

    # Quadratic concentration penalty — high-demand cells get stronger gradient
    loss = (density_map ** 2).sum()
    return loss
