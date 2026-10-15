# PIFS Scheme for HEad Pose Estimation Aimed at Faster Face Recognition (SHEEF)

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776ab?logo=python&logoColor=white)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-green?logo=creativecommons&logoColor=white)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of Bisogni et al., IEEE TBIOM 2022: partitioned iterated function systems (PIFS) fractal codes for head-pose estimation (and pose-gated face recognition).

**Repository:** https://github.com/biometric-community/pifs-scheme-for-head-pose-estimation-aimed-at-faster-face-recognition

Monorepo path: `projects/papers/pifs-scheme-for-head-pose-estimation-aimed-at-faster-face-recognition`

## Setup

```bash
bash scripts/setup_env.sh   # uses monorepo .venv
```

## Data

Requires `projects/datasets/aflw2000` (manifest + hardlinked faces from AFLW). See `DEVIATIONS.md` (D1). BIWI is optional and not shipped.

```bash
bash scripts/check_dataset_size.sh ../../datasets/aflw2000
```

## Train / Eval / Report / Predict

```bash
bash scripts/smoke.sh                 # small gallery/probe
bash scripts/train_full.sh            # size-gated full HPE protocol
bash scripts/predict.sh --image /path/to/face.jpg
```

## Layout

- `sheef/` — fractal codec, distances, AFLW2000 loader, train/eval/predict/report
- `configs/` — `default.yaml` (full), `smoke.yaml`
- `REPORT.md` — metrics + figures from real runs
- `FIDELITY_AUDIT.md`, `DEVIATIONS.md`, `SOURCE_CODE.md`

## Citation

```
@article{bisogni2022pifs,
  title={PIFS Scheme for HEad Pose Estimation Aimed at Faster Face Recognition},
  author={Bisogni, Carmen and Nappi, Michele and Pero, Chiara and Ricciardi, Stefano},
  journal={IEEE Transactions on Biometrics, Behavior, and Identity Science},
  year={2022}
}
```
