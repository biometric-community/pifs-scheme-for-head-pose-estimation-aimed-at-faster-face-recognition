"""Evaluate SHEEF HPE (paper Algorithm 1 test; MAE yaw/pitch/roll)."""

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
from sheef.distances import DISTANCES, pairwise_canberra
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
    t0 = time.time()
    mask = face_mask_128(bgr, rect)
    vec = codes_to_vector(fractal_encode(mask, domain_step=domain_step))
    return face_id, vec, np.asarray([pitch, yaw, roll], dtype=np.float64), time.time() - t0


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="SHEEF: evaluate HPE MAE")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--distance", default=None, help="override distance name")
    args = p.parse_args(argv)
    cfg = _load_cfg(args.config)

    tmpl_path = Path(cfg["paths"].get("template_dir", "outputs/templates")) / "hpe_template.npz"
    if not tmpl_path.is_file():
        raise FileNotFoundError(f"Missing template {tmpl_path}; run train first")
    data = np.load(tmpl_path)
    gallery = data["encodings"]
    gal_poses = data["poses"]  # pitch, yaw, roll
    domain_step = int(data["domain_step"]) if "domain_step" in data.files else int(cfg["model"].get("domain_step", 16))

    root = Path(cfg["data"]["root"])
    max_images = cfg["data"].get("max_images")
    gallery_size = int(cfg["data"].get("gallery_size", 750))
    seed = int(cfg["train"].get("seed", 42))
    distance_name = args.distance or cfg["eval"].get("distance", "canberra")
    max_probe = cfg["eval"].get("max_probe")
    workers = int(cfg["eval"].get("workers", cfg["train"].get("workers", 8)))

    samples = load_aflw2000_manifest(root, max_images=max_images)
    _, probe = split_gallery_probe(samples, gallery_size=gallery_size, seed=seed)
    if max_probe is not None:
        probe = probe[: int(max_probe)]

    dist_fn = DISTANCES[distance_name]
    jobs = [
        (s.face_id, str(s.path), s.rect, s.pitch, s.yaw, s.roll, domain_step) for s in probe
    ]
    abs_err = []
    correct_bin = 0
    t_enc = 0.0
    t_match = 0.0
    n_ok = 0
    t_wall = time.time()

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_encode_job, j) for j in jobs]
        for fut in as_completed(futs):
            try:
                _fid, vec, gt, enc_s = fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"skip probe: {e}")
                continue
            t_enc += enc_s
            t1 = time.time()
            if distance_name == "canberra":
                d = pairwise_canberra(vec, gallery)
            else:
                d = np.asarray([dist_fn(vec, g) for g in gallery])
            j = int(np.argmin(d))
            t_match += time.time() - t1
            err = np.abs(gal_poses[j] - gt)
            abs_err.append(err)
            if np.all(err <= 15.0):
                correct_bin += 1
            n_ok += 1
            if n_ok % 50 == 0 or n_ok == 1:
                print(f"probe {n_ok}/{len(jobs)}  wall={time.time()-t_wall:.1f}s")

    if n_ok == 0:
        raise RuntimeError("No probe images evaluated")

    E = np.stack(abs_err, axis=0)
    mae = E.mean(axis=0)
    metrics = {
        "n_probe": n_ok,
        "n_gallery": int(gallery.shape[0]),
        "distance": distance_name,
        "mae_pitch": float(mae[0]),
        "mae_yaw": float(mae[1]),
        "mae_roll": float(mae[2]),
        "mae_mean": float(mae.mean()),
        "within_15deg_all_axes": float(correct_bin / n_ok),
        "sec_encode_per_image": t_enc / n_ok,
        "sec_match_per_image": t_match / n_ok,
        "workers": workers,
        "template": str(tmpl_path),
    }
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "eval_results.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
