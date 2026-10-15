# Implementation plan — SHEEF / PIFS

Paper: Bisogni et al., *PIFS Scheme for HEad Pose Estimation Aimed at Faster Face Recognition*, IEEE TBIOM 2022 (DOI 10.1109/TBIOM.2021.3122307).

## Method inventory

| Component | Paper | Ours |
|-----------|-------|------|
| Face detect | dlib HOG+SVM | AFLW `FaceRect` (fallback Haar) |
| Face mask | 68-landmark hull (dlib) | Elliptical mask in rect |
| Resize | 128×128 | 128×128 |
| PIFS encode | Domain 8 / Range 4 → 256×6 = 1536 | 8×8 ranges / 16×16 domains → 256×6 |
| HPE | NN over pose-labeled gallery | Canberra NN (Alg. 1) |
| FR + pose filter | BIWI, ±5° step (Alg. 2) | **Not run** (BIWI absent) |
| Best distance | Canberra (Tables I–III) | Canberra default |

## Figure inventory (Results)

| Paper fig/table | Type | Our artifact |
|-----------------|------|--------------|
| Tables VI–VII MAE | bar | `outputs/figures/*/hpe_mae.svg` |
| Table VIII timing | bar | `outputs/figures/*/hpe_time.svg` |
| Fig. 4 acc vs step | FR only | skipped (needs BIWI) |

## Dataset size gate

- Required root: `projects/datasets/aflw2000` (~0.85 GiB hardlinked AFLW faces + manifest)
- Decision: **full_train** (&lt; 5 GiB)
- BIWI (~5.6 GiB) optional / missing → FR deferred (`DEVIATIONS.md`)

## GitHub repo name

`pifs-scheme-for-head-pose-estimation-aimed-at-faster-face-recognition`
