"""Multi-Leaf Collimator (MLC) sliding window leaf sequencer."""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class MLCControlPoint:
    """Aperture shape and cumulative Monitor Units (MU) at a Linac delivery control point."""
    cp_index: int
    cumulative_mu: float
    left_leaves_cm: np.ndarray   # Position of left leaf bank for each row
    right_leaves_cm: np.ndarray  # Position of right leaf bank for each row


class SlidingWindowMLCSequencer:
    """Converts 2D fluence maps into deliverable dynamic Multi-Leaf Collimator (DMLC) trajectories."""

    def __init__(
        self,
        leaf_width_cm: float = 0.5,
        min_leaf_gap_cm: float = 0.05,  # 0.5 mm minimum leaf gap
        max_leaf_speed_cm_per_mu: float = 0.05,
    ) -> None:
        self.leaf_width = float(leaf_width_cm)
        self.min_gap = float(min_leaf_gap_cm)
        self.max_speed = float(max_leaf_speed_cm_per_mu)

    def sequence_fluence_map(
        self,
        fluence_2d: np.ndarray,
        num_control_points: int = 10,
        pixel_width_cm: float = 0.5,
    ) -> List[MLCControlPoint]:
        """Sequence 2D intensity map into a series of dynamic MLC control points."""
        M, N = fluence_2d.shape
        I = np.maximum(np.asarray(fluence_2d, dtype=np.float64), 0.0)

        total_mu = float(np.max(I))
        if total_mu <= 1e-6:
            total_mu = 1.0

        # Physical coordinates across leaf travel axis (X)
        x_coords = (np.arange(N) - (N - 1) / 2.0) * pixel_width_cm
        x_min = float(x_coords[0]) - 0.5 * pixel_width_cm
        x_max = float(x_coords[-1]) + 0.5 * pixel_width_cm

        control_points: List[MLCControlPoint] = []
        fractional_mus = np.linspace(0.0, 1.0, num_control_points)

        for cp_i, frac in enumerate(fractional_mus):
            curr_mu = frac * total_mu

            # For each leaf row, calculate left and right leaf edges
            # In sliding window, left leaf sweeps following cumulative exposure
            left_pos = np.zeros(M, dtype=np.float64)
            right_pos = np.zeros(M, dtype=np.float64)

            for row in range(M):
                # Target opening: find interval where target intensity >= curr_mu
                profile = I[row, :]
                active_cols = np.where(profile > curr_mu)[0]
                if len(active_cols) > 0:
                    l_idx = active_cols[0]
                    r_idx = active_cols[-1]
                    left_pos[row] = x_coords[l_idx] - 0.5 * pixel_width_cm
                    right_pos[row] = x_coords[r_idx] + 0.5 * pixel_width_cm
                else:
                    # Park leaves closed
                    mid_x = x_min + (x_max - x_min) * frac
                    left_pos[row] = mid_x - 0.5 * self.min_gap
                    right_pos[row] = mid_x + 0.5 * self.min_gap

                # Enforce minimum gap
                if right_pos[row] - left_pos[row] < self.min_gap:
                    right_pos[row] = left_pos[row] + self.min_gap

            control_points.append(
                MLCControlPoint(
                    cp_index=cp_i,
                    cumulative_mu=round(curr_mu, 2),
                    left_leaves_cm=left_pos,
                    right_leaves_cm=right_pos,
                )
            )

        return control_points
