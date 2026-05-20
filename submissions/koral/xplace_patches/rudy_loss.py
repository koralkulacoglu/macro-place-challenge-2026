"""
Differentiable congestion-repulsion loss for Xplace GP.

v2: gradient pushes nodes AWAY from routing hot spots.

Root cause of v1 failure: scattering HPWL to net centers and minimizing sum(density^2)
produces a gradient identical to wirelength (cluster macros). v1 = wrong direction.

v2 fix: separate computation into two steps:
  1. Compute routing demand map from current positions (fully DETACHED — no autograd).
  2. Interpolate congestion value at each node's current position (WITH autograd).

Gradient of step 2: ∂loss/∂pos[i] = local congestion gradient at pos[i].
Gradient descent moves each node toward lower congestion — correct direction.
"""

import torch
import torch.nn.functional as F


def _build_demand_map(conn_node_pos, data):
    """Build RUDY demand map (call inside torch.no_grad() for detached output)."""
    device = conn_node_pos.device
    dtype  = conn_node_pos.dtype
    nbx, nby = int(data.num_bin_x), int(data.num_bin_y)
    lx, hx, ly, hy = [float(v) for v in data.die_info.tolist()]
    canvas_w = max(hx - lx, 1e-6)
    canvas_h = max(hy - ly, 1e-6)
    bin_w = canvas_w / nbx
    bin_h = canvas_h / nby

    pin_pos    = conn_node_pos[data.pin_id2node_id] + data.pin_rel_cpos
    hp_pin_pos = pin_pos[data.hyperedge_list]
    hp_x = hp_pin_pos[:, 0]
    hp_y = hp_pin_pos[:, 1]

    num_nets = int(data.hyperedge_list_end.shape[0])
    ends    = data.hyperedge_list_end
    starts  = torch.cat([torch.zeros(1, dtype=torch.long, device=device), ends[:-1]])
    lengths = ends - starts
    net_ids = torch.repeat_interleave(torch.arange(num_nets, device=device), lengths)

    gamma = 5.0 / max(canvas_w, canvas_h)

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
    hpwl   = bbox_w + bbox_h

    if data.net_mask is not None:
        hpwl = hpwl * data.net_mask.to(dtype=dtype)

    # Scatter HPWL/4 to each bbox corner — where routes actually start/end.
    # Nets with overlapping bboxes create high demand at shared corners,
    # giving a more accurate congestion map than scattering to net center.
    density_flat = torch.zeros(nbx * nby, device=device, dtype=dtype)
    corner_w = hpwl * 0.25
    for corner_x, corner_y in [(x_min, y_min), (x_min, y_max),
                                (x_max, y_min), (x_max, y_max)]:
        bx  = ((corner_x - lx) / bin_w - 0.5).clamp(0.0, nbx - 1.001)
        by_ = ((corner_y - ly) / bin_h - 0.5).clamp(0.0, nby - 1.001)
        bx0 = bx.long().clamp(0, nbx - 1)
        by0 = by_.long().clamp(0, nby - 1)
        bx1 = (bx0 + 1).clamp(0, nbx - 1)
        by1 = (by0 + 1).clamp(0, nby - 1)
        fx  = (bx  - bx0.to(dtype=dtype)).clamp(0.0, 1.0)
        fy  = (by_ - by0.to(dtype=dtype)).clamp(0.0, 1.0)
        i00 = (bx0 * nby + by0).clamp(0, nbx * nby - 1)
        i01 = (bx0 * nby + by1).clamp(0, nbx * nby - 1)
        i10 = (bx1 * nby + by0).clamp(0, nbx * nby - 1)
        i11 = (bx1 * nby + by1).clamp(0, nbx * nby - 1)
        density_flat.scatter_add_(0, i00, corner_w * (1.0 - fx) * (1.0 - fy))
        density_flat.scatter_add_(0, i01, corner_w * (1.0 - fx) * fy)
        density_flat.scatter_add_(0, i10, corner_w * fx * (1.0 - fy))
        density_flat.scatter_add_(0, i11, corner_w * fx * fy)

    return density_flat.view(nbx, nby), bin_w, bin_h, lx, ly


def rudy_congestion_loss(conn_node_pos, data):
    """
    Congestion-repulsion loss. Returns scalar for autograd.

    Gradient: ∂loss/∂pos[i] = local congestion gradient at pos[i].
    Descent moves each node from high-congestion cells to low-congestion cells.

    This is fundamentally different from v1 (which minimized total HPWL,
    equivalent to a WL gradient that CLUSTERS macros — wrong direction).
    """
    device = conn_node_pos.device
    dtype  = conn_node_pos.dtype
    nbx, nby = int(data.num_bin_x), int(data.num_bin_y)

    # ── Step 1: demand map (fully detached — constant reference for step 2) ──
    with torch.no_grad():
        density_map, bin_w, bin_h, lx, ly = _build_demand_map(conn_node_pos, data)
        mean_demand = density_map.mean().clamp(min=1e-6)
        # Normalized excess demand: 0 below average, >0 congested
        cong_map = F.relu(density_map / mean_demand - 1.0)  # [nbx, nby], no grad

    # ── Step 2: interpolate congestion at each node position (WITH autograd) ─
    # cong_map is a constant; gradient flows only through node positions.
    # ∂loss/∂x[i] = (c10-c00)*(1-fy) + (c11-c01)*fy)/bin_w
    #             = finite-diff congestion gradient → pushes node toward lower cong.
    node_x = conn_node_pos[:, 0]
    node_y = conn_node_pos[:, 1]

    bx_n  = ((node_x - lx) / bin_w - 0.5).clamp(0.0, nbx - 1.001)
    by_n  = ((node_y - ly) / bin_h - 0.5).clamp(0.0, nby - 1.001)
    bx0_n = bx_n.long().clamp(0, nbx - 1)
    by0_n = by_n.long().clamp(0, nby - 1)
    bx1_n = (bx0_n + 1).clamp(0, nbx - 1)
    by1_n = (by0_n + 1).clamp(0, nby - 1)
    fx_n  = (bx_n  - bx0_n.to(dtype=dtype)).clamp(0.0, 1.0)
    fy_n  = (by_n  - by0_n.to(dtype=dtype)).clamp(0.0, 1.0)

    cong_flat = cong_map.view(-1)  # detached constant
    c00 = cong_flat[(bx0_n * nby + by0_n).clamp(0, nbx * nby - 1)]
    c01 = cong_flat[(bx0_n * nby + by1_n).clamp(0, nbx * nby - 1)]
    c10 = cong_flat[(bx1_n * nby + by0_n).clamp(0, nbx * nby - 1)]
    c11 = cong_flat[(bx1_n * nby + by1_n).clamp(0, nbx * nby - 1)]

    cong_at_node = (c00 * (1.0 - fx_n) * (1.0 - fy_n) +
                    c01 * (1.0 - fx_n) * fy_n +
                    c10 * fx_n * (1.0 - fy_n) +
                    c11 * fx_n * fy_n)

    loss = cong_at_node.sum()
    return loss
