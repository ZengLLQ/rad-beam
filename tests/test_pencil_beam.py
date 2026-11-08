"""Unit tests for Linac geometry, PBEK kernels, and 3D pencil beam dose calculation."""

import pytest
import numpy as np
from rad_beam.dose.linac import LinacBeam
from rad_beam.dose.kernel import PolyenergeticKernel
from rad_beam.dose.pencil_beam import PencilBeamDoseEngine


def test_linac_bev_projection():
    beam = LinacBeam(beam_id=0, gantry_angle_deg=0.0, sad_cm=100.0)
    # Gantry 0 points along -Y axis in IEC standard
    # Isocenter (0,0,0) in BEV must project to (0, 0, 0)
    x_bev, y_bev, depth = beam.project_point_to_bev(np.array([0.0, 0.0, 0.0]))
    assert x_bev == pytest.approx(0.0)
    assert y_bev == pytest.approx(0.0)
    assert depth == pytest.approx(0.0)

    # Point shifted laterally by +2 cm along X
    x_bev_lat, _, _ = beam.project_point_to_bev(np.array([2.0, 0.0, 0.0]))
    assert x_bev_lat == pytest.approx(2.0)


def test_polyenergetic_kernel_pdd():
    kernel = PolyenergeticKernel(energy_mv=6.0)
    assert kernel.d_max == 1.5

    # PDD at surface d=0 must be low, peak at d_max, and attenuate deeper
    pdd_0 = float(kernel.percentage_depth_dose(0.0))
    pdd_max = float(kernel.percentage_depth_dose(1.5))
    pdd_deep = float(kernel.percentage_depth_dose(10.0))

    assert pdd_max > pdd_0
    assert pdd_max > pdd_deep
    assert pdd_deep > 0.0


def test_pencil_beam_influence_matrix():
    engine = PencilBeamDoseEngine(grid_shape=(10, 10, 8), voxel_size_cm=0.5, energy_mv=6.0)
    beam = LinacBeam(beam_id=0, gantry_angle_deg=0.0)
    b_coords = [(0.0, 0.0), (1.0, 0.0)]

    A = engine.build_influence_matrix([beam], [b_coords])
    V = 10 * 10 * 8
    assert A.shape == (V, 2)
    assert np.all(A >= 0.0)
    assert np.max(A) > 0.0
