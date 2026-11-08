"""Pencil beam energy deposition kernel and percentage depth dose (PDD) models."""

from __future__ import annotations
import math
import numpy as np


class PolyenergeticKernel:
    r"""3D Polyenergetic Pencil Beam Energy Deposition Kernel (PBEK).

    Combines primary electron transport and Compton secondary scatter:
    .. math::
        K(r, d) = A_1 \cdot \frac{1}{2\pi \sigma_1^2} e^{-r^2 / 2\sigma_1^2} + A_2 \cdot \frac{1}{2\pi \sigma_2^2} e^{-r^2 / 2\sigma_2^2}
    """

    def __init__(
        self,
        energy_mv: float = 6.0,
        ssd_cm: float = 100.0,
    ) -> None:
        self.energy = float(energy_mv)
        self.ssd = float(ssd_cm)

        # Depth of maximum dose d_max (cm) & linear attenuation mu (1/cm)
        if self.energy <= 6.0:
            self.d_max = 1.5  # 1.5 cm for 6 MV
            self.mu = 0.046   # ~3% attenuation per cm in water
            self.beta_bu = 2.4
        else:
            self.d_max = 2.3  # 2.3 cm for 10-15 MV
            self.mu = 0.038
            self.beta_bu = 1.8

        # Dual Gaussian radial scatter kernel parameters (cm)
        self.sigma1_0 = 0.22  # 2.2 mm core primary electron pencil
        self.alpha1 = 0.015
        self.sigma2_0 = 1.6   # 16 mm broad Compton scatter
        self.alpha2 = 0.06
        self.weight_primary = 0.88
        self.weight_scatter = 0.12

    def percentage_depth_dose(self, depth_cm: np.ndarray | float) -> np.ndarray:
        r"""Compute Central-Axis Percentage Depth Dose (PDD) normalized to 1.0 at d_max."""
        d = np.maximum(np.asarray(depth_cm, dtype=np.float64), 0.0)
        # Inverse square correction
        inv_sq = ((self.ssd + self.d_max) / (self.ssd + d)) ** 2

        # Build-up and exponential attenuation
        bu_factor = 1.0 - np.exp(-self.beta_bu * d)
        att_factor = np.exp(-self.mu * np.maximum(d - self.d_max, 0.0))

        pdd = inv_sq * bu_factor * att_factor
        return np.maximum(pdd, 0.0)

    def radial_kernel_2d(self, r_cm: np.ndarray | float, depth_cm: float) -> np.ndarray:
        """Evaluate radial scatter kernel value at distance r [cm] from central ray at given depth."""
        d = max(float(depth_cm), 0.1)
        sigma1 = self.sigma1_0 + self.alpha1 * d
        sigma2 = self.sigma2_0 + self.alpha2 * d

        r_sq = np.asarray(r_cm, dtype=np.float64) ** 2

        g1 = (self.weight_primary / (2.0 * np.pi * (sigma1 ** 2))) * np.exp(-r_sq / (2.0 * (sigma1 ** 2)))
        g2 = (self.weight_scatter / (2.0 * np.pi * (sigma2 ** 2))) * np.exp(-r_sq / (2.0 * (sigma2 ** 2)))

        return g1 + g2
