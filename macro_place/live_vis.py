"""Live, animated visualization of a placer working in real time.

Unlike ``visualize_placement`` in :mod:`macro_place.utils` (which renders a
static PNG *after* ``place()`` finishes), this opens an interactive window and
updates it continuously while the placer runs.  A placer fires a progress
callback (see ``LiveVisualizer.update``); the visualizer repaints persistent
matplotlib artists on the main thread — no threads, no video saving.

Three panels, dark-neon styling:
    1. Placement   — macros sliding around the canvas, ports on the border
    2. Density     — live density heatmap (when grids are supplied)
    3. Congestion  — live smoothed-congestion heatmap

Frame dict consumed by ``update`` (only ``positions``/``phase`` are required)::

    {
        "positions":       np.ndarray [N, 2] of macro centers,
        "phase":           str,                       # e.g. "GP", "LAHC"
        "iteration":       int,
        "proxy" / "wl" / "density" / "congestion": float,
        "best":            float,
        "elapsed":         float,                      # seconds
        "density_grid":    np.ndarray [rows, cols] | None,
        "congestion_grid": np.ndarray [rows, cols] | None,
    }

This is a *native* feature — it needs a GUI backend and will not display inside
the headless Docker dev flow.
"""

from __future__ import annotations

import os
import sys
import time
from typing import Optional

import numpy as np

# Dark-neon palette
_BG = "#0b0e14"
_FG = "#e5e7eb"
_ACCENT = "#22d3ee"
_C_HARD = "#22d3ee"      # hard movable macros
_C_HARD_EDGE = "#67e8f9"
_C_FIXED = "#ff4dd2"     # fixed macros
_C_FIXED_EDGE = "#ffa3ec"
_C_SOFT = "#0e7490"      # soft macros (faint)
_C_SOFT_EDGE = "#22d3ee"
_C_PORT = "#39ff14"      # I/O ports
_C_BORDER = "#3b4252"
_C_OUTLINE = "#67e8f9"   # hard-macro outline drawn over heatmaps

_NON_GUI_BACKENDS = {"agg", "pdf", "ps", "svg", "template", "cairo", "pgf"}


def _ensure_interactive_backend() -> bool:
    """Return True if an interactive (GUI) matplotlib backend is active.

    If the active backend is non-interactive *and* the user did not pin one via
    ``MPLBACKEND``, try to upgrade to a GUI backend.  An explicit ``MPLBACKEND``
    is always respected (lets callers force headless rendering for tests/Docker).
    """
    import matplotlib

    if matplotlib.get_backend().lower() not in _NON_GUI_BACKENDS:
        return True
    if os.environ.get("MPLBACKEND"):
        return False  # user pinned a non-interactive backend on purpose
    for candidate in ("TkAgg", "QtAgg", "Qt5Agg", "MacOSX"):
        try:
            matplotlib.use(candidate, force=True)
            return True
        except Exception:
            continue
    return False


class LiveVisualizer:
    """Holds a live matplotlib figure and repaints it from ``update`` frames."""

    def __init__(self, benchmark, min_interval_s: float = 0.1):
        self.benchmark = benchmark
        self.min_interval_s = float(min_interval_s)
        self.enabled = False
        self.interactive = False
        self._last_draw = 0.0

        try:
            import matplotlib  # noqa: F401
        except ImportError:
            print("[live] matplotlib not installed; live view disabled.", file=sys.stderr)
            return
        self.interactive = _ensure_interactive_backend()
        if not self.interactive:
            print(
                "[live] No interactive matplotlib backend (rendering headlessly, "
                "no window will appear). For the live view, run natively — not in "
                "Docker — with Tk or Qt installed.",
                file=sys.stderr,
            )
        import matplotlib.pyplot as plt

        self._plt = plt
        # Cache drawing data as numpy (positions arrive per-frame; the rest is static).
        self.n_hard = int(benchmark.num_hard_macros)
        self.n_macros = int(benchmark.num_macros)
        self.sizes = benchmark.macro_sizes.cpu().numpy().astype(float)
        self.half = self.sizes / 2.0
        self.fixed = benchmark.macro_fixed.cpu().numpy().astype(bool)
        self.movable = (~benchmark.macro_fixed).cpu().numpy().astype(bool)
        self.cw = float(benchmark.canvas_width)
        self.ch = float(benchmark.canvas_height)
        self.grid_rows = int(benchmark.grid_rows)
        self.grid_cols = int(benchmark.grid_cols)
        self.extent = (0.0, self.cw, 0.0, self.ch)

        self._build_figure()
        self.enabled = True

    # ── figure construction (persistent artists created once) ─────────────
    def _build_figure(self):
        plt = self._plt
        from matplotlib.patches import Rectangle

        plt.ion()
        self.fig, self.axes = plt.subplots(1, 3, figsize=(19, 6.8))
        self.fig.patch.set_facecolor(_BG)

        init = self.benchmark.macro_positions.cpu().numpy().astype(float)

        # Panel 0 — placement
        ax = self.axes[0]
        self._style_axis(ax, f"{self.benchmark.name} · placement")
        ax.add_patch(Rectangle((0, 0), self.cw, self.ch, fill=False,
                               edgecolor=_C_BORDER, linewidth=1.5))
        self.macro_patches = []
        for i in range(self.n_macros):
            x, y = init[i]
            w, h = self.sizes[i]
            is_soft = i >= self.n_hard
            if self.fixed[i]:
                fc, ec, a, ls, lw = _C_FIXED, _C_FIXED_EDGE, 0.9, "solid", 0.6
            elif is_soft:
                fc, ec, a, ls, lw = _C_SOFT, _C_SOFT_EDGE, 0.22, "dashed", 0.4
            else:
                fc, ec, a, ls, lw = _C_HARD, _C_HARD_EDGE, 0.85, "solid", 0.6
            rect = Rectangle((x - w / 2, y - h / 2), w, h, facecolor=fc, alpha=a,
                             edgecolor=ec, linewidth=lw, linestyle=ls,
                             zorder=2 if is_soft else 4)
            ax.add_patch(rect)
            self.macro_patches.append(rect)
        ports = self.benchmark.port_positions.cpu().numpy()
        if ports.shape[0] > 0:
            ax.scatter(ports[:, 0], ports[:, 1], s=10, c=_C_PORT, zorder=6,
                       edgecolors="none")

        # Panels 1 & 2 — heatmaps
        zero = np.zeros((self.grid_rows, self.grid_cols))
        self.dens_im = self._make_heatmap(self.axes[1], zero, "magma",
                                          f"{self.benchmark.name} · density")
        self.cong_im = self._make_heatmap(self.axes[2], zero, "hot",
                                          f"{self.benchmark.name} · congestion")
        self.dens_outlines = self._add_hard_outlines(self.axes[1], init)
        self.cong_outlines = self._add_hard_outlines(self.axes[2], init)
        self._warm = [
            self._warming_text(self.axes[1]),
            self._warming_text(self.axes[2]),
        ]

        # HUD text
        self.hud = self.fig.text(
            0.5, 0.975, "", ha="center", va="top", color=_FG,
            family="monospace", fontsize=11,
        )
        self.fig.tight_layout(rect=(0, 0, 1, 0.95))
        self.fig.canvas.draw_idle()
        self.fig.canvas.flush_events()

    def _style_axis(self, ax, title):
        ax.set_facecolor(_BG)
        ax.set_xlim(0, self.cw)
        ax.set_ylim(0, self.ch)
        ax.set_aspect("equal")
        ax.set_title(title, color=_ACCENT, fontsize=11, family="monospace")
        for spine in ax.spines.values():
            spine.set_color(_C_BORDER)
        ax.tick_params(colors="#6b7280", labelsize=7)

    def _make_heatmap(self, ax, grid, cmap, title):
        self._style_axis(ax, title)
        im = ax.imshow(grid, origin="lower", extent=self.extent, aspect="equal",
                       cmap=cmap, vmin=0.0, vmax=1.0, zorder=0,
                       interpolation="nearest")
        return im

    def _add_hard_outlines(self, ax, init):
        from matplotlib.patches import Rectangle

        patches = []
        for i in range(self.n_hard):
            x, y = init[i]
            w, h = self.sizes[i]
            ec = _C_FIXED_EDGE if self.fixed[i] else _C_OUTLINE
            rect = Rectangle((x - w / 2, y - h / 2), w, h, fill=False,
                             edgecolor=ec, linewidth=0.5, zorder=3, alpha=0.7)
            ax.add_patch(rect)
            patches.append(rect)
        return patches

    def _warming_text(self, ax):
        return ax.text(0.5, 0.5, "warming up…", transform=ax.transAxes,
                       ha="center", va="center", color="#9ca3af",
                       family="monospace", fontsize=12, zorder=10)

    # ── per-frame update ──────────────────────────────────────────────────
    def update(self, frame: dict, force: bool = False):
        """Repaint from a frame dict.  Safe to call at high frequency — it
        throttles internally and never raises into the placer."""
        if not self.enabled:
            return
        now = time.time()
        if not force and (now - self._last_draw) < self.min_interval_s:
            return
        try:
            self._render(frame)
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self._last_draw = time.time()
        except Exception as e:  # never let the viz kill a long placer run
            print(f"[live] render error (continuing): {e}", file=sys.stderr)
            self.enabled = False

    def _render(self, frame: dict):
        pos = np.asarray(frame["positions"], dtype=float)
        # Move macros (skip fixed — they never move)
        for i in range(min(len(self.macro_patches), pos.shape[0])):
            if not self.movable[i]:
                continue
            self.macro_patches[i].set_xy((pos[i, 0] - self.half[i, 0],
                                          pos[i, 1] - self.half[i, 1]))
        for i in range(min(len(self.dens_outlines), pos.shape[0])):
            if not self.movable[i]:
                continue
            xy = (pos[i, 0] - self.half[i, 0], pos[i, 1] - self.half[i, 1])
            self.dens_outlines[i].set_xy(xy)
            self.cong_outlines[i].set_xy(xy)

        self._update_heatmap(self.dens_im, frame.get("density_grid"),
                             self._warm[0], percentile=None)
        self._update_heatmap(self.cong_im, frame.get("congestion_grid"),
                             self._warm[1], percentile=99.0)
        self.hud.set_text(self._hud_text(frame))

    def _update_heatmap(self, im, grid, warm_text, percentile):
        if grid is None:
            return
        grid = np.asarray(grid, dtype=float)
        im.set_data(grid)
        if percentile is not None:
            pos = grid[grid > 0]
            vmax = float(np.percentile(pos, percentile)) if pos.size else 1.0
        else:
            vmax = float(grid.max())
        im.set_clim(0.0, max(vmax, 1e-9))
        if warm_text.get_visible():
            warm_text.set_visible(False)

    def _hud_text(self, frame: dict) -> str:
        phase = frame.get("phase", "?")
        it = frame.get("iteration")
        elapsed = frame.get("elapsed")
        line1 = f"{self.benchmark.name}  |  phase={phase}"
        if it is not None:
            line1 += f"  it={int(it):,}"
        if elapsed is not None:
            line1 += f"  t={elapsed:.0f}s"

        def fmt(key):
            v = frame.get(key)
            return f"{v:.4f}" if isinstance(v, (int, float)) else "—"

        line2 = (f"proxy={fmt('proxy')}   wl={fmt('wl')}  "
                 f"dens={fmt('density')}  cong={fmt('congestion')}   "
                 f"best={fmt('best')}")
        return line1 + "\n" + line2

    # ── teardown ──────────────────────────────────────────────────────────
    def finish(self, placement=None, costs: Optional[dict] = None):
        """Render a final frame and block until the user closes the window."""
        if not self.enabled:
            return
        try:
            if placement is not None:
                final = {
                    "positions": np.asarray(placement, dtype=float),
                    "phase": "DONE",
                }
                if costs:
                    final.update({
                        "proxy": costs.get("proxy_cost"),
                        "wl": costs.get("wirelength_cost"),
                        "density": costs.get("density_cost"),
                        "congestion": costs.get("congestion_cost"),
                        "best": costs.get("proxy_cost"),
                    })
                self.update(final, force=True)
            if self.interactive:
                print("[live] Done — close the window to exit.", flush=True)
                self._plt.ioff()
                self._plt.show()
            else:
                self._plt.close(self.fig)
        except Exception as e:
            print(f"[live] finish error: {e}", file=sys.stderr)
