"""
Differentiable RUDY (Routing Utilization Density Estimation) congestion loss for Xplace GP.

Computes a smooth, autograd-differentiable RUDY density map from current macro positions
using numerically-stable log-sum-exp bounding boxes. Bypasses gpugr (which crashes on IBM).

Injected into calculator.py after the WL gradient when args.rudy_weight > 0.
"""

import torch
import torch.nn.functional as F


def rudy_congestion_loss(conn_node_pos, data):
    """
    Differentiable RUDY congestion loss. Returns scalar for autograd.

    RUDY: routing demand per net ≈ HPWL.
    Demand is scattered to grid cells at the net center via bilinear interpolation.
    Loss = sum(density_map^2) — penalizes routing demand concentration.
    """
    device = conn_node_pos.device
    dtype  = conn_node_pos.dtype
    nbx, nby = int(data.num_bin_x), int(data.num_bin_y)
    lx, hx, ly, hy = [float(v) for v in data.die_info.tolist()]
    canvas_w = max(hx - lx, 1e-6)
    canvas_h = max(hy - ly, 1e-6)
    bin_w = canvas_w / nbx
    bin_h = canvas_h / nby

    # Absolute pin positions: [num_pins, 2]
    pin_pos    = conn_node_pos[data.pin_id2node_id] + data.pin_rel_cpos
    hp_pin_pos = pin_pos[data.hyperedge_list]   # [total_hp_pins, 2]
    hp_x = hp_pin_pos[:, 0]
    hp_y = hp_pin_pos[:, 1]

    # net_ids[i] = net index for each entry in hyperedge_list
    num_nets = int(data.hyperedge_list_end.shape[0])
    ends    = data.hyperedge_list_end
    starts  = torch.cat([torch.zeros(1, dtype=torch.long, device=device), ends[:-1]])
    lengths = ends - starts
    net_ids = torch.repeat_interleave(torch.arange(num_nets, device=device), lengths)

    # Numerically stable LSE bounding box.
    # max_x[k] ≈ (1/γ) log Σ_j exp(γ x_j)  for pins j in net k.
    # Stability: clamp γ*x to [-85, 85] (float32 safe: exp(85) ≈ 7e36 < 3.4e38).
    gamma = 5.0 / max(canvas_w, canvas_h)   # small enough to avoid overflow

    def lse_max(vals, ids, n):
        z = (gamma * vals).clamp(-85.0, 85.0)
        s = torch.zeros(n, device=device, dtype=dtype).scatter_add_(0, ids, torch.exp(z))
        return s.clamp(min=1e-30).log() / gamma

    def lse_min(vals, ids, n):
        z = (-gamma * vals).clamp(-85.0, 85.0)
        s = torch.zeros(n, device=device, dtype=dtype).scatter_add_(0, ids, torch.exp(z))
        return -(s.clamp(min=1e-30).log()) / gamma

    x_max = lse_max(hp_x, net_ids, num_nets)
    x_min = lse_min(hp_x, net_ids, num_nets)
    y_max = lse_max(hp_y, net_ids, num_nets)
    y_min = lse_min(hp_y, net_ids, num_nets)

    bbox_w = (x_max - x_min).clamp(min=bin_w)
    bbox_h = (y_max - y_min).clamp(min=bin_h)
    hpwl   = bbox_w + bbox_h   # routing demand per net

    # Apply net mask
    if data.net_mask is not None:
        hpwl = hpwl * data.net_mask.to(dtype=dtype)

    # Scatter demand to grid via bilinear at net center
    cx  = (x_max + x_min) * 0.5
    cy  = (y_max + y_min) * 0.5
    bx  = ((cx - lx) / bin_w - 0.5).clamp(0.0, nbx - 1.001)
    by_ = ((cy - ly) / bin_h - 0.5).clamp(0.0, nby - 1.001)

    bx0 = bx.long().clamp(0, nbx - 1)
    by0 = by_.long().clamp(0, nby - 1)
    bx1 = (bx0 + 1).clamp(0, nbx - 1)
    by1 = (by0 + 1).clamp(0, nby - 1)
    fx  = (bx  - bx0.to(dtype=dtype)).clamp(0.0, 1.0)
    fy  = (by_ - by0.to(dtype=dtype)).clamp(0.0, 1.0)

    # Safe flat indices
    i00 = (bx0 * nby + by0).clamp(0, nbx * nby - 1)
    i01 = (bx0 * nby + by1).clamp(0, nbx * nby - 1)
    i10 = (bx1 * nby + by0).clamp(0, nbx * nby - 1)
    i11 = (bx1 * nby + by1).clamp(0, nbx * nby - 1)

    density_flat = torch.zeros(nbx * nby, device=device, dtype=dtype)
    density_flat.scatter_add_(0, i00, hpwl * (1.0 - fx) * (1.0 - fy))
    density_flat.scatter_add_(0, i01, hpwl * (1.0 - fx) * fy)
    density_flat.scatter_add_(0, i10, hpwl * fx * (1.0 - fy))
    density_flat.scatter_add_(0, i11, hpwl * fx * fy)

    density_map = density_flat.view(nbx, nby)

    # Normalize by mean and apply quadratic overflow penalty
    mean_demand = density_map.mean().detach().clamp(min=1e-6)
    overflow = F.relu(density_map / mean_demand - 1.0)
    loss = (overflow ** 2).sum()
    return loss
