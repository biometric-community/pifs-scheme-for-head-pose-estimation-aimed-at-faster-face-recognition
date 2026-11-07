"""REPORT.md, the committed output of the full run."""

from __future__ import annotations

import re

import pytest


def metric(report: str, name: str) -> float:
    match = re.search(rf"\| {re.escape(name)} \| ([\d.]+) \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return float(match.group(1))


def test_mean_mae_is_the_mean_of_the_three_angles(report):
    angles = [metric(report, f"MAE {axis} (°)") for axis in ("pitch", "yaw", "roll")]
    assert metric(report, "MAE mean (°)") == pytest.approx(sum(angles) / 3.0, abs=0.002)


def test_every_image_was_used(report):
    assert metric(report, "Gallery size") + metric(report, "Probe size") >= 1990


def test_published_values(report, targets):
    if targets["published_values"] is None:
        pytest.skip("paper tables not recorded in tests/paper_targets.yaml")
    for axis, expected in targets["published_values"].items():
        assert metric(report, f"MAE {axis} (°)") == pytest.approx(expected, abs=1.0)
