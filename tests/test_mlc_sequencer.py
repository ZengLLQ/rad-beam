"""Unit tests for Multi-Leaf Collimator (MLC) sequencer."""

import pytest
import numpy as np
from rad_beam.planning.mlc_sequencer import SlidingWindowMLCSequencer


def test_sliding_window_sequencer():
    sequencer = SlidingWindowMLCSequencer(leaf_width_cm=0.5, min_leaf_gap_cm=0.1)

    # 4 rows x 5 columns fluence map
    fluence = np.array([
        [0.0, 10.0, 20.0, 10.0, 0.0],
        [5.0, 15.0, 25.0, 15.0, 5.0],
        [5.0, 15.0, 25.0, 15.0, 5.0],
        [0.0, 10.0, 20.0, 10.0, 0.0],
    ])

    control_points = sequencer.sequence_fluence_map(fluence, num_control_points=6)
    assert len(control_points) == 6

    # Verify minimum leaf gap constraint across all control points and rows
    for cp in control_points:
        assert len(cp.left_leaves_cm) == 4
        assert len(cp.right_leaves_cm) == 4
        gaps = cp.right_leaves_cm - cp.left_leaves_cm
        assert np.all(gaps >= 0.099)
