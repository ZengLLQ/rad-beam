"""Radiation therapy inverse planning objective functions and analytic gradients."""

from __future__ import annotations
from typing import Dict, List
import numpy as np


class PlanObjective:
    r"""Inverse planning multi-criteria optimization objective.

    .. math::
        f(\vec{w}) = \frac{w_{ptv}}{N_{ptv}} \sum_{i \in PTV} (D_i - D_{rx})^2 + \sum_{k} \frac{w_k}{N_k} \sum_{j \in OAR_k} [\max(D_j - D_{tol, k}, 0)]^2
    """

    def __init__(self) -> None:
        self.ptv_indices: np.ndarray | None = None
        self.d_prescribed: float = 60.0  # e.g. 60 Gy
        self.w_ptv: float = 10.0

        self.oar_structures: Dict[str, Dict[str, object]] = {}

    def set_ptv(self, voxel_indices: np.ndarray, prescribed_dose_gy: float, weight: float = 10.0) -> None:
        """Define Planning Target Volume (PTV) and prescription dose."""
        self.ptv_indices = np.asarray(voxel_indices, dtype=int)
        self.d_prescribed = float(prescribed_dose_gy)
        self.w_ptv = float(weight)

    def add_oar(self, name: str, voxel_indices: np.ndarray, max_dose_tolerance_gy: float, weight: float = 5.0) -> None:
        """Add Organ-at-Risk (OAR) dose constraint."""
        self.oar_structures[name] = {
            "indices": np.asarray(voxel_indices, dtype=int),
            "d_tol": float(max_dose_tolerance_gy),
            "weight": float(weight),
        }

    def evaluate(self, dose_vector: np.ndarray) -> float:
        """Evaluate total penalty cost f(D)."""
        cost = 0.0

        # PTV penalty
        if self.ptv_indices is not None and len(self.ptv_indices) > 0:
            d_ptv = dose_vector[self.ptv_indices]
            n_ptv = len(self.ptv_indices)
            cost += (self.w_ptv / n_ptv) * float(np.sum((d_ptv - self.d_prescribed) ** 2))

        # OAR penalties
        for oar in self.oar_structures.values():
            idx = oar["indices"]
            if len(idx) == 0:
                continue
            d_oar = dose_vector[idx]
            d_tol = oar["d_tol"]
            weight = oar["weight"]
            excess = np.maximum(d_oar - d_tol, 0.0)
            cost += (weight / len(idx)) * float(np.sum(excess ** 2))

        return float(cost)

    def gradient(self, dose_vector: np.ndarray, influence_matrix: np.ndarray) -> np.ndarray:
        r"""Compute exact analytic gradient w.r.t beamlet weights: \nabla_w f = A^T \nabla_D f."""
        V = len(dose_vector)
        grad_D = np.zeros(V, dtype=np.float64)

        # PTV gradient
        if self.ptv_indices is not None and len(self.ptv_indices) > 0:
            d_ptv = dose_vector[self.ptv_indices]
            n_ptv = len(self.ptv_indices)
            grad_D[self.ptv_indices] = (2.0 * self.w_ptv / n_ptv) * (d_ptv - self.d_prescribed)

        # OAR gradient
        for oar in self.oar_structures.values():
            idx = oar["indices"]
            if len(idx) == 0:
                continue
            d_oar = dose_vector[idx]
            d_tol = oar["d_tol"]
            weight = oar["weight"]
            excess = np.maximum(d_oar - d_tol, 0.0)
            grad_D[idx] += (2.0 * weight / len(idx)) * excess

        # Backproject through influence matrix A^T * grad_D
        return np.dot(influence_matrix.T, grad_D)
