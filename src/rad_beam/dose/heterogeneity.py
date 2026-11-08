"""Tissue heterogeneity correction and CT Hounsfield Unit (HU) electron density calibration."""

from __future__ import annotations
import numpy as np


class TissueHeterogeneityCorrector:
    """Calculates relative electron density and effective radiological depth across heterogeneous CT tissue."""

    @staticmethod
    def hu_to_relative_electron_density(hu: np.ndarray | float) -> np.ndarray:
        r"""Bi-linear stoichiometric calibration curve from CT Hounsfield Units (HU) to electron density relative to water.

        .. math::
            \rho_{rel} = 1.0 + 0.001 \cdot HU \quad (HU \le 0)
            \rho_{rel} = 1.0 + 0.00055 \cdot HU \quad (HU > 0)
        """
        hu_arr = np.asarray(hu, dtype=np.float64)
        rho = np.empty_like(hu_arr)

        mask_low = hu_arr <= 0.0
        rho[mask_low] = 1.0 + 0.001 * hu_arr[mask_low]
        rho[~mask_low] = 1.0 + 0.00055 * hu_arr[~mask_low]

        # Bound density to physical limits (Air: 0.001 to Cortical Bone: 2.1)
        return np.clip(rho, 0.001, 2.2)

    @classmethod
    def compute_radiological_depth(
        cls,
        density_grid: np.ndarray,
        voxel_size_cm: float,
        axis: int = 1,
    ) -> np.ndarray:
        """Compute cumulative effective radiological depth d_eff [cm] along beam ray traversal axis."""
        # Cumulative sum of density along axis multiplied by physical voxel step size
        return np.cumsum(density_grid, axis=axis) * voxel_size_cm
