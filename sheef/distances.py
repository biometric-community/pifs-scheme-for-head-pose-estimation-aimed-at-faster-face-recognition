"""Distance metrics over fractal code vectors (paper Tables I–III)."""

from __future__ import annotations

import numpy as np


def canberra(a: np.ndarray, b: np.ndarray) -> float:
    """Canberra distance (paper Eq. 8) — best for HPE and FR on both sets."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denom = np.abs(a) + np.abs(b)
    # avoid 0/0
    mask = denom > 0
    if not np.any(mask):
        return 0.0
    return float(np.sum(np.abs(a[mask] - b[mask]) / denom[mask]))


def euclidean(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)))


def cityblock(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def chebyshev(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64))))


def hamming_quantized(a: np.ndarray, b: np.ndarray, bins: int = 32) -> float:
    """Hamming on coarsely quantized codes (proxy for paper Hamming on codec)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    # per-dimension min/max fragile; use fixed scale from typical codec ranges
    qa = np.clip((a * 10).astype(np.int32), -bins, bins)
    qb = np.clip((b * 10).astype(np.int32), -bins, bins)
    return float(np.mean(qa != qb))


DISTANCES = {
    "canberra": canberra,
    "euclidean": euclidean,
    "cityblock": cityblock,
    "chebyshev": chebyshev,
    "hamming": hamming_quantized,
}


def pairwise_canberra(query: np.ndarray, gallery: np.ndarray) -> np.ndarray:
    """Vectorized Canberra distances from one query to all gallery rows."""
    q = np.asarray(query, dtype=np.float64)[None, :]
    g = np.asarray(gallery, dtype=np.float64)
    denom = np.abs(q) + np.abs(g)
    num = np.abs(q - g)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(denom > 0, num / denom, 0.0)
    return ratio.sum(axis=1)
