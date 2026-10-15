"""AFLW2000 manifest loader (pose-labeled faces)."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple


@dataclass
class Aflw2000Sample:
    face_id: int
    path: Path
    yaw: float
    pitch: float
    roll: float
    rect: Optional[Tuple[int, int, int, int]]


def load_aflw2000_manifest(root: str | Path, max_images: Optional[int] = None) -> List[Aflw2000Sample]:
    root = Path(root)
    man = root / "extracted" / "manifest.csv"
    if not man.is_file():
        raise FileNotFoundError(
            f"Missing {man}. Build from AFLW via the monorepo dataset helper "
            "(projects/datasets/aflw2000/extracted/manifest.csv)."
        )
    samples: List[Aflw2000Sample] = []
    with man.open() as f:
        for row in csv.DictReader(f):
            rect = None
            try:
                if row.get("x") not in (None, "", "None"):
                    rect = (int(float(row["x"])), int(float(row["y"])), int(float(row["w"])), int(float(row["h"])))
            except (TypeError, ValueError):
                rect = None
            samples.append(
                Aflw2000Sample(
                    face_id=int(row["face_id"]),
                    path=Path(row["path"]),
                    yaw=float(row["yaw_deg"]),
                    pitch=float(row["pitch_deg"]),
                    roll=float(row["roll_deg"]),
                    rect=rect,
                )
            )
            if max_images is not None and len(samples) >= max_images:
                break
    if not samples:
        raise RuntimeError(f"No samples in {man}")
    return samples


def split_gallery_probe(
    samples: Sequence[Aflw2000Sample],
    gallery_size: int = 750,
    seed: int = 42,
) -> Tuple[List[Aflw2000Sample], List[Aflw2000Sample]]:
    """Paper AFLW2000 template size ≈ 750 encodings."""
    import numpy as np

    idx = np.arange(len(samples))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    g = max(1, min(gallery_size, len(samples) - 1))
    gallery = [samples[i] for i in idx[:g]]
    probe = [samples[i] for i in idx[g:]]
    if not probe:
        probe = gallery[-1:]
        gallery = gallery[:-1]
    return gallery, probe
