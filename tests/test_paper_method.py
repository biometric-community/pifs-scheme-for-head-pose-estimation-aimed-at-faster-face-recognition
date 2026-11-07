"""Fractal codec, distances and the gallery protocol, checked on controlled inputs."""

from __future__ import annotations

import numpy as np
import pytest

from sheef.data.aflw2000 import Aflw2000Sample, load_aflw2000_manifest, split_gallery_probe
from sheef.distances import DISTANCES, canberra, chebyshev, cityblock, euclidean, pairwise_canberra
from sheef.fractal import _fit, codes_to_vector, fractal_encode


def test_canberra_follows_its_definition():
    a, b = np.array([1.0, -2.0, 0.0]), np.array([3.0, 2.0, 0.0])
    assert canberra(a, b) == pytest.approx(2.0 / 4.0 + 4.0 / 4.0)


def test_vectorised_canberra_matches_the_scalar_form():
    rng = np.random.RandomState(0)
    query, gallery = rng.normal(size=12), rng.normal(size=(5, 12))
    assert np.allclose(pairwise_canberra(query, gallery), [canberra(query, g) for g in gallery])


def test_other_distances_follow_their_definitions():
    a, b = np.array([0.0, 3.0]), np.array([4.0, 0.0])
    assert euclidean(a, b) == 5.0
    assert cityblock(a, b) == 7.0
    assert chebyshev(a, b) == 4.0


def test_distance_of_a_vector_to_itself_is_zero():
    v = np.random.RandomState(1).normal(size=8)
    for name, fn in DISTANCES.items():
        assert fn(v, v) == 0.0, name


def test_affine_fit_recovers_contrast_and_brightness():
    d = np.random.RandomState(2).rand(8, 8)
    s, o, mse = _fit(d, 0.5 * d + 0.1)
    assert (s, o) == (pytest.approx(0.5), pytest.approx(0.1))
    assert mse == pytest.approx(0.0, abs=1e-12)


def test_contrast_is_kept_contractive():
    d = np.random.RandomState(3).rand(8, 8)
    assert abs(_fit(d, 5.0 * d)[0]) < 1.0


def test_flat_domain_falls_back_to_the_range_mean():
    s, o, _ = _fit(np.ones((8, 8)), np.full((8, 8), 0.3))
    assert s == 0.0 and o == pytest.approx(0.3)


@pytest.fixture(scope="module")
def image():
    rng = np.random.RandomState(4)
    return np.kron(rng.rand(16, 16), np.ones((8, 8))) * 255.0


def test_encoding_has_the_paper_length(image, targets):
    codes = fractal_encode(image, domain_step=64)
    assert codes.shape == (256, 6)
    assert codes_to_vector(codes).size == targets["encoding_dim"]


def test_encoding_is_deterministic(image):
    assert np.array_equal(fractal_encode(image, domain_step=64), fractal_encode(image, domain_step=64))


def test_encoder_rejects_other_image_sizes():
    with pytest.raises(ValueError):
        fractal_encode(np.zeros((64, 64)))


def samples(n: int) -> list:
    return [Aflw2000Sample(i, f"{i}.jpg", 0.0, 0.0, 0.0, None) for i in range(n)]


def test_gallery_and_probe_are_disjoint():
    gallery, probe = split_gallery_probe(samples(100), gallery_size=30)
    assert len(gallery) == 30 and len(probe) == 70
    assert not {s.face_id for s in gallery} & {s.face_id for s in probe}


def test_configured_gallery_matches_the_paper_template(default_cfg, targets):
    assert default_cfg["data"]["gallery_size"] == targets["gallery_size"]


def test_manifest_poses_are_read_in_degrees(tmp_path):
    root = tmp_path / "extracted"
    root.mkdir()
    (root / "manifest.csv").write_text(
        "face_id,path,yaw_deg,pitch_deg,roll_deg,x,y,w,h\n1,a.jpg,10.5,-3,2,1,2,3,4\n"
    )
    sample = load_aflw2000_manifest(tmp_path)[0]
    assert (sample.yaw, sample.pitch, sample.roll) == (10.5, -3.0, 2.0)
    assert sample.rect == (1, 2, 3, 4)
