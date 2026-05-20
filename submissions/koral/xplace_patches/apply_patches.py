"""
Applies in-place text patches to Xplace source files to add differentiable RUDY support.
Run once inside the Docker image after copying rudy_loss.py to /opt/xplace/src/core/.
"""

from pathlib import Path

XPLACE = Path("/opt/xplace")


def patch(path: Path, old: str, new: str):
    txt = path.read_text()
    if old not in txt:
        print(f"  [SKIP] {path.name}: marker not found (already patched?)")
        return
    path.write_text(txt.replace(old, new, 1))
    print(f"  [OK]   {path.name} patched")


# ── 1. calculator.py: inject RUDY gradient after WL grad ────────────────────
patch(
    XPLACE / "src/calculator.py",
    old="        mov_node_pos.grad[mov_lhs:mov_rhs] += conn_node_grad_by_wl[mov_lhs:mov_rhs]",
    new=(
        "        mov_node_pos.grad[mov_lhs:mov_rhs] += conn_node_grad_by_wl[mov_lhs:mov_rhs]\n"
        "\n"
        "        # ── Differentiable RUDY congestion gradient ──────────────────────────────\n"
        "        # Trigger: overflow < 0.35 (density converging, macros roughly placed).\n"
        "        # Recompute gradient every K=50 iters to keep overhead ~2s/seed.\n"
        "        _rudy_K = 50\n"
        "        _rudy_ovfl_list = getattr(ps.recorder, 'overflow', [])\n"
        "        _rudy_triggered = len(_rudy_ovfl_list) > 0 and _rudy_ovfl_list[-1] < 0.08\n"
        "        if getattr(ps, 'use_rudy', False) and _rudy_triggered:\n"
        "            if not hasattr(ps, '_rudy_grad_cache'): ps._rudy_grad_cache = None; ps._rudy_cache_iter = -999; ps._rudy_first = True\n"
        "            if ps.iter - ps._rudy_cache_iter >= _rudy_K or ps._rudy_grad_cache is None:\n"
        "                from src.core.rudy_loss import rudy_congestion_loss\n"
        "                _cp = conn_node_pos.detach().clone().requires_grad_(True)\n"
        "                _rl = rudy_congestion_loss(_cp, data)\n"
        "                _rl.backward()\n"
        "                ps._rudy_grad_cache = _cp.grad[:mov_rhs - mov_lhs].detach().clone() if _cp.grad is not None else None\n"
        "                ps._rudy_cache_iter = ps.iter\n"
        "                if getattr(ps, '_rudy_first', True) or ps.iter % 200 == 0:\n"
        "                    import sys as _sys; _sys.stderr.write(f'  [RUDY] iter={ps.iter} ovfl={_rudy_ovfl_list[-1]:.3f} loss={_rl.item():.4f}\\n'); _sys.stderr.flush()\n"
        "                    ps._rudy_first = False\n"
        "            if ps._rudy_grad_cache is not None:\n"
        "                _rw = float(ps.rudy_weight) if not hasattr(ps.rudy_weight, '__len__') else float(ps.rudy_weight.mean())\n"
        "                mov_node_pos.grad[mov_lhs:mov_rhs] += ps._rudy_grad_cache * _rw"
    ),
)

# ── 2. main.py: add --rudy_weight and --rudy_start_iter args ────────────────
patch(
    XPLACE / "main.py",
    old="    parser.add_argument('--congest_weight', type=float, default=0, help='the weight of congested force')",
    new=(
        "    parser.add_argument('--congest_weight', type=float, default=0, help='the weight of congested force')\n"
        "    parser.add_argument('--rudy_weight', type=float, default=0.0,\n"
        "                        help='weight of differentiable RUDY congestion gradient')\n"
        "    parser.add_argument('--rudy_start_iter', type=int, default=6000,\n"
        "                        help='iteration to start RUDY gradient (after WL/density settle, overflow<0.2)')"
    ),
)

# ── 3. param_scheduler.py: add use_rudy / rudy_weight fields ────────────────
patch(
    XPLACE / "src/param_scheduler.py",
    old="        self.congest_weight = args.congest_weight",
    new=(
        "        self.congest_weight = args.congest_weight\n"
        "        self.use_rudy   = getattr(args, 'rudy_weight', 0.0) > 0\n"
        "        self.rudy_weight = getattr(args, 'rudy_weight', 0.0)"
    ),
)

print("Done.")
