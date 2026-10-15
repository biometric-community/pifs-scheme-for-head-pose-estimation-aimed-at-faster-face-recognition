# Deviations — SHEEF / PIFS

| ID | Topic | Paper | Ours | Rationale |
|----|-------|-------|------|-----------|
| D1 | AFLW2000 source | AFLW2000 / AFLW2000-3D (Zhu et al.) | First 2000 resolvable AFLW faces with classic `FacePose` (rad→deg), hardlinked under `projects/datasets/aflw2000` | Official AFLW2000-3D not staged; local AFLW provides pose GT |
| D2 | Face detect / landmarks | dlib HOG + Kazemi–Sullivan 68-pt mask | Dataset `FaceRect` + elliptical mask (Haar fallback) | `dlib` not in shared env; rects exist for AFLW |
| D3 | PIFS block geometry | Domain 8 / Range 4 stated; 256×6 codec | Non-overlapping 8×8 ranges + 16×16 domains (pooled), 4 isometries, `domain_step=16` | Match 1536-D length with tractable CPU search |
| D4 | BIWI + face recognition | BIWI Kinect (~15k frames, 20 IDs); Alg. 2 pose-gated FR | Not executed | BIWI payload absent / ≥5 GiB; HPE-only full protocol on AFLW2000 |
| D5 | Metric inventory | Tables I–III many distances | Default Canberra; helpers for Euclidean/cityblock/Chebyshev/Hamming-proxy | Canberra is paper’s best for HPE+FR |

No fabricated metrics: `REPORT.md` only from real `outputs/logs/**/eval_results.json`.
