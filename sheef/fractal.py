"""Jacquin-style PIFS fractal encoding (paper Sec. III).

Paper: 128x128 face; Domain 8 / Range 4; codec 256 x 6 -> 1536-D.
We emit 256 codes from non-overlapping 8x8 ranges with 16x16 domains
(average-pooled to 8x8) so the vector length matches. See DEVIATIONS.md.
"""

from __future__ import annotations

import numpy as np

# 4 isometries (subset of 8) for speed; documented in DEVIATIONS.
_ISOS = (
    lambda a: a,
    lambda a: np.rot90(a, 1),
    lambda a: np.rot90(a, 2),
    lambda a: np.fliplr(a),
)


def _fit(d: np.ndarray, r: np.ndarray) -> tuple[float, float, float]:
    d = d.ravel()
    r = r.ravel()
    n = d.size
    sum_d = d.sum()
    sum_r = r.sum()
    sum_dd = np.dot(d, d)
    sum_dr = np.dot(d, r)
    denom = n * sum_dd - sum_d * sum_d
    if abs(denom) < 1e-8:
        s = 0.0
        o = sum_r / n
    else:
        s = (n * sum_dr - sum_d * sum_r) / denom
        s = float(np.clip(s, -0.99, 0.99))
        o = (sum_r - s * sum_d) / n
    err = r - (s * d + o)
    return float(s), float(o), float(np.dot(err, err) / n)


def fractal_encode(gray_128: np.ndarray, domain_step: int = 16) -> np.ndarray:
    """Encode 128x128 grayscale -> (256, 6): dom_r, dom_c, iso, s, o, mse."""
    if gray_128.shape != (128, 128):
        raise ValueError(f"expected 128x128, got {gray_128.shape}")
    img = gray_128.astype(np.float64)
    rs, ds = 8, 16
    n_r = 128 // rs
    codes = np.zeros((n_r * n_r, 6), dtype=np.float64)

    max_o = 128 - ds
    origins = [(i, j) for i in range(0, max_o + 1, domain_step) for j in range(0, max_o + 1, domain_step)]
    pooled = []
    for di, dj in origins:
        block = img[di : di + ds, dj : dj + ds]
        pooled.append(block.reshape(8, 2, 8, 2).mean(axis=(1, 3)))

    # Precompute isometries of each domain
    dom_bank = []
    for p in pooled:
        dom_bank.append([iso(p) for iso in _ISOS])

    idx = 0
    for ri in range(n_r):
        for rj in range(n_r):
            rang = img[ri * rs : (ri + 1) * rs, rj * rs : (rj + 1) * rs]
            best_mse = 1e18
            best = (0, 0, 0, 0.0, 0.0, 0.0)
            for (di, dj), isos in zip(origins, dom_bank):
                for iso_id, dom in enumerate(isos):
                    s, o, mse = _fit(dom, rang)
                    if mse < best_mse:
                        best_mse = mse
                        best = (di, dj, iso_id, s, o, mse)
            codes[idx] = best
            idx += 1
    return codes


def codes_to_vector(codes: np.ndarray) -> np.ndarray:
    return codes.astype(np.float64).reshape(-1)
