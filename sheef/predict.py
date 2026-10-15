"""Predict pitch/yaw/roll for a single image via SHEEF template."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import yaml

from sheef.distances import pairwise_canberra
from sheef.fractal import codes_to_vector, fractal_encode
from sheef.preprocess import face_mask_128


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="SHEEF: predict head pose")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--image", required=True)
    p.add_argument("--rect", nargs=4, type=int, default=None, metavar=("X", "Y", "W", "H"))
    args = p.parse_args(argv)
    cfg = yaml.safe_load(Path(args.config).read_text())
    tmpl = Path(cfg["paths"].get("template_dir", "outputs/templates")) / "hpe_template.npz"
    data = np.load(tmpl)
    gallery = data["encodings"]
    poses = data["poses"]
    domain_step = int(data["domain_step"]) if "domain_step" in data.files else 16

    bgr = cv2.imread(args.image)
    if bgr is None:
        raise FileNotFoundError(args.image)
    rect = tuple(args.rect) if args.rect else None
    vec = codes_to_vector(fractal_encode(face_mask_128(bgr, rect), domain_step=domain_step))
    j = int(np.argmin(pairwise_canberra(vec, gallery)))
    pitch, yaw, roll = [float(x) for x in poses[j]]
    out = {"image": args.image, "pitch": pitch, "yaw": yaw, "roll": roll, "gallery_index": j}
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
