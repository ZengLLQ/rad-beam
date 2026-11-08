"""Medical Linear Accelerator (Linac) beam geometry and coordinate transformations."""

from __future__ import annotations
import math
import numpy as np


class LinacBeam:
    r"""Medical linear accelerator megavoltage photon beam specification.

    Handles Beam's-Eye-View (BEV) to patient 3D isocentric coordinate transformations.
    """

    def __init__(
        self,
        beam_id: int,
        gantry_angle_deg: float,
        couch_angle_deg: float = 0.0,
        energy_mv: float = 6.0,
        sad_cm: float = 100.0,
        isocenter_cm: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        self.beam_id = int(beam_id)
        self.gantry_angle = float(gantry_angle_deg)
        self.couch_angle = float(couch_angle_deg)
        self.energy_mv = float(energy_mv)
        self.sad = float(sad_cm)
        self.isocenter = np.array(isocenter_cm, dtype=np.float64)

        # Transformation rotation matrix from Patient (X: Right->Left, Y: Ant->Post, Z: Inf->Sup)
        # to Beam's-Eye-View (X_bev, Y_bev, Z_bev pointing along central axis toward target)
        theta_g = math.radians(self.gantry_angle)
        # Standard IEC 61217 gantry rotation around Z-axis:
        cos_g = math.cos(theta_g)
        sin_g = math.sin(theta_g)
        # Central beam direction unit vector
        self.central_axis_dir = np.array([-sin_g, -cos_g, 0.0], dtype=np.float64)
        # Source position in patient coordinates
        self.source_pos = self.isocenter - self.sad * self.central_axis_dir

    def project_point_to_bev(self, point_patient: np.ndarray) -> tuple[float, float, float]:
        r"""Project 3D patient point into Beam's-Eye-View coordinates (x_bev, y_bev, depth_cm)."""
        pt = np.asarray(point_patient, dtype=np.float64)
        vec_from_source = pt - self.source_pos

        # Depth along central ray
        depth = float(np.dot(vec_from_source, self.central_axis_dir)) - self.sad

        theta_g = math.radians(self.gantry_angle)
        cos_g = math.cos(theta_g)
        sin_g = math.sin(theta_g)

        # Lateral and longitudinal coordinates in isocentric plane (SAD=100cm)
        rel_iso = pt - self.isocenter
        x_bev = float(rel_iso[0] * cos_g - rel_iso[1] * sin_g)
        y_bev = float(rel_iso[2])  # Along Z (cranio-caudal)

        return x_bev, y_bev, depth
