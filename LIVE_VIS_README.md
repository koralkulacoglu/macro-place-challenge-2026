# Live Placement Visualization (`--live`)

Watch the placer optimize a chip floorplan in real time: a dark-neon, 3-panel
view — **placement** (macros sliding around, ports on the border) | live
**density heatmap** | live **congestion heatmap** — with a HUD showing phase,
iteration, and the proxy / wirelength / density / congestion costs.

## Setup (one-time)

```bash
git submodule update --init external/MacroPlacement   # benchmark testcases (required)
uv sync                                                # installs torch (CPU), matplotlib, numpy
```

## Run

```bash
uv run evaluate submissions/idk/placer.py -b ibm01 --live
```

## Notes

- **Run it natively** on a normal Windows desktop session — not inside Docker
  (Docker has no display). matplotlib uses the bundled Tk (`tkagg`) backend, so
  no extra GUI install is needed.
- **No GPU required** — runs on CPU. A full run uses the default ~55-min budget:
  hard-macro motion (GP) is early, heatmaps light up once the FastEvaluator
  builds, and LAHC is the long polishing tail.
- Close the window to exit, or press `Ctrl+C` to stop early.
- `--live` runs one benchmark at a time (pick it with `-b <name>`, e.g. `ibm01`).
