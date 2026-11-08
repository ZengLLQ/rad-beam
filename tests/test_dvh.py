"""Unit tests for Dose-Volume Histogram (DVH) calculation."""

import pytest
import numpy as np
from rad_beam.clinical.dvh import DoseVolumeHistogram


def test_dvh_monotonicity_and_metrics():
    # Synthetic dose distribution: 100 voxels with dose uniformly spaced between 40 and 60 Gy
    roi_doses = np.linspace(40.0, 60.0, 100)
    dvh = DoseVolumeHistogram(roi_doses)
    roi_indices = np.arange(100)

    d_bins, vol_pct = dvh.cumulative_dvh(roi_indices)

    # Monotonicity check: vol_pct must be monotonically non-increasing
    diffs = np.diff(vol_pct)
    assert np.all(diffs <= 0.0)

    # Boundary checks: at dose < 40 Gy, 100% volume
    assert dvh.get_vx(roi_indices, 30.0) == 100.0
    # At dose > 60 Gy, 0% volume
    assert dvh.get_vx(roi_indices, 65.0) == 0.0

    # Specific metrics
    m = dvh.compute_structure_metrics(roi_indices)
    assert m["d_min_gy"] == pytest.approx(40.0)
    assert m["d_max_gy"] == pytest.approx(60.0)
    assert m["d_mean_gy"] == pytest.approx(50.0)
    assert m["d50_gy"] == pytest.approx(50.0, abs=1.0)
