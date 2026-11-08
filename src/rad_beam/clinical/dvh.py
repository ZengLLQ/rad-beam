"""Dose-Volume Histogram (DVH) calculation and dosimetric parameters."""

from __future__ import annotations
from typing import Dict, Tuple
import numpy as np


class DoseVolumeHistogram:
    """Calculates cumulative and differential Dose-Volume Histograms (DVH)."""

    def __init__(self, dose_array: np.ndarray, num_bins: int = 150) -> None:
        self.doses = np.asarray(dose_array, dtype=np.float64)
        self.max_dose = float(np.max(self.doses)) if len(self.doses) > 0 else 0.0
        self.num_bins = int(num_bins)

        # Dose bin edges up to 105% of max dose
        self.bin_edges = np.linspace(0.0, max(self.max_dose * 1.05, 1.0), self.num_bins + 1)
        self.bin_centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:])

    def cumulative_dvh(self, voxel_indices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Compute cumulative DVH (dose_bins_gy, volume_pct) for a structure."""
        if len(voxel_indices) == 0:
            return self.bin_centers, np.zeros_like(self.bin_centers)

        roi_doses = self.doses[voxel_indices]
        N_roi = len(roi_doses)

        # Cumulative volume percentage receiving >= D
        # Sort doses descending
        sorted_doses = np.sort(roi_doses)
        vol_pct = np.zeros_like(self.bin_centers)

        for i, d in enumerate(self.bin_centers):
            # Count voxels with dose >= d
            count = np.sum(sorted_doses >= d)
            vol_pct[i] = (count / N_roi) * 100.0

        return self.bin_centers, vol_pct

    def get_dx(self, voxel_indices: np.ndarray, volume_pct: float) -> float:
        r"""Compute dose D_x delivered to at least x percent of volume."""
        if len(voxel_indices) == 0:
            return 0.0
        roi_doses = self.doses[voxel_indices]
        # x% volume receives at least D_x -> percentile(100 - x)
        return float(np.percentile(roi_doses, 100.0 - volume_pct))

    def get_vx(self, voxel_indices: np.ndarray, dose_threshold_gy: float) -> float:
        """Compute volume percentage V_x receiving at least dose_threshold_gy."""
        if len(voxel_indices) == 0:
            return 0.0
        roi_doses = self.doses[voxel_indices]
        return float(np.sum(roi_doses >= dose_threshold_gy) / len(roi_doses)) * 100.0

    def compute_structure_metrics(self, voxel_indices: np.ndarray) -> Dict[str, float]:
        """Extract key dosimetric parameters for structure."""
        if len(voxel_indices) == 0:
            return {}
        roi_doses = self.doses[voxel_indices]
        return {
            "d_min_gy": round(float(np.min(roi_doses)), 2),
            "d_mean_gy": round(float(np.mean(roi_doses)), 2),
            "d_max_gy": round(float(np.max(roi_doses)), 2),
            "d98_gy": round(self.get_dx(voxel_indices, 98.0), 2),
            "d95_gy": round(self.get_dx(voxel_indices, 95.0), 2),
            "d50_gy": round(self.get_dx(voxel_indices, 50.0), 2),
            "d2_gy": round(self.get_dx(voxel_indices, 2.0), 2),
        }
