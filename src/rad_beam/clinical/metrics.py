"""Clinical radiation oncology plan quality indices (CI, HI)."""

from __future__ import annotations
import numpy as np
from rad_beam.clinical.dvh import DoseVolumeHistogram


class PlanQualityMetrics:
    """Computes ICRU and RTOG plan conformity and homogeneity metrics."""

    @staticmethod
    def homogeneity_index(dvh: DoseVolumeHistogram, ptv_indices: np.ndarray) -> float:
        r"""Compute ICRU-83 Homogeneity Index (HI).

        .. math::
            HI = \frac{D_{2\%} - D_{98\%}}{D_{50\%}}
        Values close to 0 denote ideal dose homogeneity inside the PTV.
        """
        metrics = dvh.compute_structure_metrics(ptv_indices)
        d2 = metrics["d2_gy"]
        d98 = metrics["d98_gy"]
        d50 = metrics["d50_gy"]
        return float((d2 - d98) / max(d50, 1e-6))

    @staticmethod
    def conformity_index(
        full_dose_grid: np.ndarray,
        ptv_mask_3d: np.ndarray,
        reference_isodose_gy: float,
    ) -> float:
        r"""Compute Paddick Conformity Index (CI).

        .. math::
            CI = \frac{V_{PTV, RI}^2}{V_{PTV} \cdot V_{RI}}
        where:
        - $V_{PTV, RI}$: Target volume receiving at least reference isodose.
        - $V_{PTV}$: Total target volume.
        - $V_{RI}$: Total patient volume receiving at least reference isodose.
        CI = 1.0 denotes perfect target conformity without normal tissue overdose.
        """
        isodose_mask = full_dose_grid >= reference_isodose_gy
        v_ptv_ri = np.sum(isodose_mask & ptv_mask_3d)
        v_ptv = np.sum(ptv_mask_3d)
        v_ri = np.sum(isodose_mask)

        if v_ptv == 0 or v_ri == 0:
            return 0.0

        paddick_ci = float((v_ptv_ri ** 2) / (v_ptv * v_ri))
        return round(paddick_ci, 3)
