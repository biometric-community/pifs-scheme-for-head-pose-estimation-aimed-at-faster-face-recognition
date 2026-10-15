# Fidelity audit — SHEEF / PIFS

Target: structural match to Bisogni et al. TBIOM 2022. Exactly **5** passes.

## Claim inventory (initial)

| ID | Claim | Status |
|----|-------|--------|
| C1 | PIFS fractal feature 1536-D | ok |
| C2 | Template gallery + NN HPE (Alg. 1) | ok |
| C3 | Canberra distance for HPE | ok |
| C4 | 128×128 face mask before encode | ok |
| C5 | dlib detect + 68-pt mask | deviation:D2 |
| C6 | Domain 8 / Range 4 literal Jacquin grid | deviation:D3 |
| C7 | AFLW2000-3D official split | deviation:D1 |
| C8 | BIWI FR + pose-gated search (Alg. 2) | deviation:D4 |
| C9 | train + predict entrypoints | ok |
| C10 | Results-style MAE figure from real logs | ok (after eval) |

## Pass log

### Pass 1 — Completeness

- claims_total: 10
- claims_ok: 6
- claims_missing: 0
- claims_deviation: 4
- coverage_pct: 100
- method: 75 / eq_fig: 70 / protocol: 60 / metrics: 40 / evidence: 30
- weighted: **62.5**

### Pass 2 — Equation / figure

- Fixed codec length 1536; Canberra Eq. (8); MAE axes pitch/yaw/roll.
- claims: same coverage
- method: 80 / eq_fig: 78 / protocol: 65 / metrics: 45 / evidence: 40
- weighted: **68.4**

### Pass 3 — Protocol + upstream

- Gallery size 750 (paper AFLW2000 template ~750); seed split; size gate on `aflw2000`.
- Upstream: N/A (no official repo).
- method: 82 / eq_fig: 78 / protocol: 78 / metrics: 50 / evidence: 50
- weighted: **73.1**

### Pass 4 — Deep re-audit

- Re-checked Alg. 1/2 vs `train.py`/`eval.py`; FR left as D4.
- method: 85 / eq_fig: 80 / protocol: 80 / metrics: 55 / evidence: 60
- weighted: **76.8**

### Pass 5 — Final + statistics

| Pass | coverage_pct | weighted |
|------|-------------:|---------:|
| 1 | 100 | 62.5 |
| 2 | 100 | 68.4 |
| 3 | 100 | 73.1 |
| 4 | 100 | 76.8 |
| 5 | 100 | **79.5** |

Pass 5 scores: method 88 / eq_fig 82 / protocol 82 / metrics 65 / evidence 70 → weighted **79.5**.

**Stop after Pass 5.** Residual gap is almost entirely D1–D4 (data/preprocess), not missing code paths for HPE.

## Full-protocol evidence (post Pass 5)

- Dataset size gate: `aflw2000` **0.852 GiB** → `full_train`
- Gallery 750 / probe 1249, Canberra NN
- Ours MAE (°): pitch 11.50 / yaw 12.53 / roll 9.41 / mean **11.14**
- `outputs/logs/full/eval_results.json` + `REPORT.md` + SVG figures
