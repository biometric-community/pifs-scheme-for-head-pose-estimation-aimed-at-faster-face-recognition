# REPORT — SHEEF / PIFS head-pose estimation

Reproduction of Bisogni et al., *PIFS Scheme for HEad Pose Estimation Aimed at Faster Face Recognition*, IEEE TBIOM 2022.

## Summary (ours)

| Metric | Value |
|--------|------:|
| Gallery size | 750 |
| Probe size | 1249 |
| Distance | canberra |
| MAE pitch (°) | 11.499 |
| MAE yaw (°) | 12.526 |
| MAE roll (°) | 9.405 |
| MAE mean (°) | 11.143 |
| Within 15° (all axes) | 0.470 |
| Encode s/img | 4.9999 |
| Match s/img | 0.3084 |

Train summary: gallery encodings=750, dim=1536, wall_s=464.300252199173.

## Figures

### HPE MAE (paper Tables VI–VII style)

![HPE MAE](outputs/figures/full/hpe_mae.svg)

### Timing

![HPE time](outputs/figures/full/hpe_time.svg)

## Notes

- Metrics are from **our** AFLW2000 protocol (classic AFLW `FacePose`), not paper table numbers.
- BIWI face recognition is not reported — see `DEVIATIONS.md`.

## Regenerate

```bash
bash scripts/report.sh
```
