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
        "        # ── Differentiable RUDY congestion gradient (no GPU router needed) ──\n"
        "        if getattr(ps, 'use_rudy', False) and ps.iter >= getattr(args, 'rudy_start_iter', 1000):\n"
        "            from src.core.rudy_loss import rudy_congestion_loss\n"
        "            _cp = conn_node_pos.detach().clone().requires_grad_(True)\n"
        "            _rl = rudy_congestion_loss(_cp, data)\n"
        "            _rl.backward()\n"
        "            if _cp.grad is not None:\n"
        "                _rw = ps.rudy_weight if hasattr(ps.rudy_weight, 'item') else ps.rudy_weight\n"
        "                mov_node_pos.grad[mov_lhs:mov_rhs] += _cp.grad[:mov_rhs - mov_lhs] * _rw\n"
        "                if ps.iter % 200 == 0:\n"
        "                    print(f'  [RUDY] iter={ps.iter} loss={_rl.item():.4f} rw={float(_rw):.4f}')"
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
        "    parser.add_argument('--rudy_start_iter', type=int, default=1000,\n"
        "                        help='iteration to start RUDY gradient (after WL/density settle)')"
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
