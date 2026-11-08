"""Projected Gradient Descent (PGD) optimizer for beamlet intensity map optimization."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from rad_beam.planning.objective import PlanObjective


class FluenceMapOptimizer:
    r"""Solves constrained non-negative quadratic inverse planning problem:

    .. math::
        \min_{\vec{w} \ge 0} f(A \vec{w})
    via Projected Gradient Descent (PGD).
    """

    def __init__(
        self,
        max_iterations: int = 100,
        learning_rate: float = 0.05,
        tol: float = 1e-4,
    ) -> None:
        self.max_iter = int(max_iterations)
        self.lr = float(learning_rate)
        self.tol = float(tol)

    def optimize(
        self,
        influence_matrix: np.ndarray,
        objective: PlanObjective,
        initial_weights: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, List[float]]:
        """Run projected gradient descent optimization.

        Returns:
            (optimal_weights, optimal_dose_vector, cost_history)
        """
        A = np.asarray(influence_matrix, dtype=np.float64)
        V, B = A.shape

        if initial_weights is None:
            # Initialize with weights scaled to prescription dose
            d_target = objective.d_prescribed if hasattr(objective, "d_prescribed") else 60.0
            if hasattr(objective, "ptv_indices") and objective.ptv_indices is not None and len(objective.ptv_indices) > 0:
                sum_per_voxel = float(np.mean(np.sum(A[objective.ptv_indices, :], axis=1)))
                scale = d_target / max(sum_per_voxel, 0.01)
            else:
                scale = 10.0
            w = np.ones(B, dtype=np.float64) * scale
        else:
            w = np.maximum(np.asarray(initial_weights, dtype=np.float64), 0.0)

        # Spectral radius estimation for adaptive step size
        # Lipschitz constant L approx max singular value of A^T A
        w_step = self.lr

        cost_history: List[float] = []

        for it in range(self.max_iter):
            dose = np.dot(A, w)
            cost = objective.evaluate(dose)
            cost_history.append(cost)

            # Gradient step
            grad_w = objective.gradient(dose, A)

            # Barzilai-Borwein or gradient norm clipping
            grad_norm = np.linalg.norm(grad_w)
            if grad_norm < self.tol:
                break

            # Adaptive step
            step = min(w_step, 1.0 / (grad_norm + 1e-6))
            w_new = np.maximum(w - step * grad_w, 0.0)

            # Check relative cost reduction
            dose_new = np.dot(A, w_new)
            cost_new = objective.evaluate(dose_new)
            if cost_new < cost:
                w = w_new
                w_step *= 1.05
            else:
                w_step *= 0.5

        final_dose = np.dot(A, w)
        return w, final_dose, cost_history
