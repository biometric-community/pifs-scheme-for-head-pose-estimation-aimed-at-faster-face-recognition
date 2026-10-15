"""Generate REPORT.md + SVG figures from real train/eval JSON (dev-plot)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from sheef.plot_style import BAR_COLOR, FIGSIZE, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="SHEEF report")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--logs", default=None)
    p.add_argument("--figdir", default=None)
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    log_dir = Path(args.logs or cfg["paths"]["log_dir"])
    fig_dir = Path(args.figdir or cfg["paths"]["figure_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)

    eval_path = log_dir / "eval_results.json"
    train_path = log_dir / "train_summary.json"
    if not eval_path.is_file():
        Path("REPORT.md").write_text(
            "# REPORT — SHEEF / PIFS\n\n"
            "**BLOCKED:** missing `eval_results.json`. Run train → eval on AFLW2000 first.\n"
        )
        print("BLOCKED: no eval_results.json")
        return

    metrics = json.loads(eval_path.read_text())
    train = json.loads(train_path.read_text()) if train_path.is_file() else {}

    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    labels = ["Pitch", "Yaw", "Roll", "Mean"]
    vals = [metrics["mae_pitch"], metrics["mae_yaw"], metrics["mae_roll"], metrics["mae_mean"]]
    bars = ax.bar(labels, vals, color=BAR_COLOR, width=0.55, edgecolor="#002080")
    ax.set_ylabel("MAE (degrees)")
    ax.set_title("SHEEF HPE MAE on AFLW2000 (ours)")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.2f}", ha="center", va="bottom", fontsize=11)
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    mae_svg = fig_dir / "hpe_mae.svg"
    fig.savefig(mae_svg, format="svg")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.bar(
        ["Encode", "Match"],
        [metrics["sec_encode_per_image"], metrics["sec_match_per_image"]],
        color=list(MULTI_SERIES_COLORS[:2]),
        width=0.5,
        edgecolor="#002080",
    )
    ax.set_ylabel("Seconds")
    ax.set_title("Per-image HPE time (ours)")
    apply_style(ax, grid_axis="y")
    fig.tight_layout()
    time_svg = fig_dir / "hpe_time.svg"
    fig.savefig(time_svg, format="svg")
    plt.close(fig)

    report = f"""# REPORT — SHEEF / PIFS head-pose estimation

Reproduction of Bisogni et al., *PIFS Scheme for HEad Pose Estimation Aimed at Faster Face Recognition*, IEEE TBIOM 2022.

## Summary (ours)

| Metric | Value |
|--------|------:|
| Gallery size | {metrics['n_gallery']} |
| Probe size | {metrics['n_probe']} |
| Distance | {metrics['distance']} |
| MAE pitch (°) | {metrics['mae_pitch']:.3f} |
| MAE yaw (°) | {metrics['mae_yaw']:.3f} |
| MAE roll (°) | {metrics['mae_roll']:.3f} |
| MAE mean (°) | {metrics['mae_mean']:.3f} |
| Within 15° (all axes) | {metrics['within_15deg_all_axes']:.3f} |
| Encode s/img | {metrics['sec_encode_per_image']:.4f} |
| Match s/img | {metrics['sec_match_per_image']:.4f} |

Train summary: gallery encodings={train.get('n_gallery')}, dim={train.get('encoding_dim')}, wall_s={train.get('seconds')}.

## Figures

### HPE MAE (paper Tables VI–VII style)

![HPE MAE]({mae_svg.as_posix()})

### Timing

![HPE time]({time_svg.as_posix()})

## Notes

- Metrics are from **our** AFLW2000 protocol (classic AFLW `FacePose`), not paper table numbers.
- BIWI face recognition is not reported — see `DEVIATIONS.md`.

## Regenerate

```bash
bash scripts/report.sh
```
"""
    Path("REPORT.md").write_text(report)
    print(f"Wrote REPORT.md and figures under {fig_dir}")


if __name__ == "__main__":
    main()
