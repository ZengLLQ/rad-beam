"""Unit tests for tissue heterogeneity correction and CT calibration."""

import pytest
import numpy as np
from rad_beam.dose.heterogeneity import TissueHeterogeneityCorrector


def test_hu_to_electron_density():
    # Water: HU = 0 -> rho = 1.0
    rho_water = float(TissueHeterogeneityCorrector.hu_to_relative_electron_density(0.0))
    assert rho_water == pytest.approx(1.0)

    # Air: HU = -1000 -> rho ~ 0.001
    rho_air = float(TissueHeterogeneityCorrector.hu_to_relative_electron_density(-1000.0))
    assert rho_air == pytest.approx(0.001, abs=0.01)

    # Dense Bone: HU = 1500 -> rho ~ 1.825
    rho_bone = float(TissueHeterogeneityCorrector.hu_to_relative_electron_density(1500.0))
    assert 1.6 < rho_bone < 2.1


def test_radiological_depth():
    # 5 voxels of lung (rho = 0.25) followed by 5 voxels of bone (rho = 1.8)
    density_line = np.array([0.25, 0.25, 0.25, 0.25, 0.25, 1.8, 1.8, 1.8, 1.8, 1.8])
    density_grid = density_line.reshape(1, 10, 1)

    d_eff = TissueHeterogeneityCorrector.compute_radiological_depth(density_grid, voxel_size_cm=1.0, axis=1)

    # After 5 cm of lung: 5 * 0.25 = 1.25 cm radiological depth
    assert d_eff[0, 4, 0] == pytest.approx(1.25)
    # Total depth after 10 cm: 1.25 + 5 * 1.8 = 10.25 cm
    assert d_eff[0, 9, 0] == pytest.approx(10.25)
