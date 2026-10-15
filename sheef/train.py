"""Build SHEEF fractal template (paper Algorithm 1 model creation)."""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np
import yaml

from sheef.data import load_aflw2000_manifest, split_gallery_probe
from sheef.fractal import codes_to_vector, fractal_encode
from sheef.preprocess import face_mask_128


def _load_cfg(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _encode_job(payload: tuple) -> tuple:
    face_id, path, rect, pitch, yaw, roll, domain_step = payload
    bgr = cv2.imread(path)
    if bgr is None:
        raise FileNotFoundError(path)
    mask = face_mask_128(bgr, rect)
    vec = codes_to_vector(fractal_encode(mask, domain_step=domain_step))
    return face_id, vec, pitch, yaw, roll


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="SHEEF: build fractal HPE template")
    p.add_argument("--config", default="configs/default.yaml")
    args = p.parse_args(argv)
    cfg = _load_cfg(args.config)

    root = Path(cfg["data"]["root"])
    max_images = cfg["data"].get("max_images")
    gallery_size = int(cfg["data"].get("gallery_size", 750))
    seed = int(cfg["train"].get("seed", 42))
    domain_step = int(cfg["model"].get("domain_step", 16))
    workers = int(cfg["train"].get("workers", 8))
    log_dir = Path(cfg["paths"]["log_dir"])
    tmpl_dir = Path(cfg["paths"].get("template_dir", "outputs/templates"))
    log_dir.mkdir(parents=True, exist_ok=True)
    tmpl_dir.mkdir(parents=True, exist_ok=True)

    samples = load_aflw2000_manifest(root, max_images=max_images)
    gallery, probe = split_gallery_probe(samples, gallery_size=gallery_size, seed=seed)

    jobs = [
        (s.face_id, str(s.path), s.rect, s.pitch, s.yaw, s.roll, domain_step) for s in gallery
    ]
    vectors = []
    poses = []
    face_ids = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_encode_job, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                face_id, vec, pitch, yaw, roll = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"skip gallery: {e}")
                continue
            vectors.append(vec)
            poses.append([pitch, yaw, roll])
            face_ids.append(face_id)
            done += 1
            if done % 25 == 0 or done == 1:
                elapsed = time.time() - t0
                print(f"gallery {done}/{len(jobs)}  {done / max(elapsed, 1e-6):.2f} img/s")

    if not vectors:
        raise RuntimeError("No gallery encodings produced")

    G = np.stack(vectors, axis=0)
    P = np.asarray(poses, dtype=np.float64)
    ids = np.asarray(face_ids, dtype=np.int64)
    np.savez_compressed(
        tmpl_dir / "hpe_template.npz",
        encodings=G,
        poses=P,
        face_ids=ids,
        probe_face_ids=np.asarray([s.face_id for s in probe], dtype=np.int64),
        gallery_size=len(G),
        domain_step=domain_step,
        seed=seed,
    )

    summary = {
        "n_gallery": int(G.shape[0]),
        "n_probe_planned": len(probe),
        "encoding_dim": int(G.shape[1]),
        "domain_step": domain_step,
        "workers": workers,
        "seconds": time.time() - t0,
        "template": str(tmpl_dir / "hpe_template.npz"),
        "config": args.config,
    }
    (log_dir / "train_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
