"""Unit tests for inverse planning objective and projected gradient descent optimizer."""

import pytest
import numpy as np
from rad_beam.planning.objective import PlanObjective
from rad_beam.planning.optimizer import FluenceMapOptimizer


def test_objective_gradient():
    obj = PlanObjective()
    ptv_idx = np.array([0, 1, 2])
    oar_idx = np.array([3, 4])
    obj.set_ptv(ptv_idx, prescribed_dose_gy=50.0, weight=10.0)
    obj.add_oar("TestOAR", oar_idx, max_dose_tolerance_gy=20.0, weight=5.0)

    # 5 voxels, 2 beamlets
    A = np.array([
        [1.0, 0.5],
        [0.8, 0.6],
        [1.1, 0.4],
        [0.2, 0.9],
        [0.1, 0.8],
    ])
    w = np.array([30.0, 20.0])
    dose = np.dot(A, w)

    cost = obj.evaluate(dose)
    grad = obj.gradient(dose, A)

    assert cost > 0.0
    assert len(grad) == 2

    # Numerical gradient check
    eps = 1e-6
    for i in range(2):
        w_pert = w.copy()
        w_pert[i] += eps
        dose_pert = np.dot(A, w_pert)
        cost_pert = obj.evaluate(dose_pert)
        grad_num = (cost_pert - cost) / eps
        assert grad[i] == pytest.approx(grad_num, rel=1e-3)


def test_fluence_map_optimizer():
    obj = PlanObjective()
    ptv_idx = np.array([0, 1])
    obj.set_ptv(ptv_idx, prescribed_dose_gy=40.0)

    A = np.array([
        [2.0, 1.0],
        [1.8, 1.2],
    ])
    optimizer = FluenceMapOptimizer(max_iterations=50, learning_rate=0.1)
    w_opt, dose_opt, history = optimizer.optimize(A, obj, initial_weights=np.array([5.0, 5.0]))

    # Weights must be non-negative
    assert np.all(w_opt >= 0.0)
    # Cost must decrease
    assert history[-1] < history[0]
    # Target dose should be close to 40 Gy
    assert np.allclose(dose_opt[ptv_idx], 40.0, rtol=0.10)
