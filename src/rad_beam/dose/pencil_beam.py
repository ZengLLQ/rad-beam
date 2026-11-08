"""3D Pencil Beam Convolution dose calculation engine."""

from __future__ import annotations
from typing import List, Tuple
import numpy as np
from rad_beam.dose.linac import LinacBeam
from rad_beam.dose.kernel import PolyenergeticKernel
from rad_beam.dose.heterogeneity import TissueHeterogeneityCorrector


class PencilBeamDoseEngine:
    r"""Calculates 3D absorbed dose distribution D(x, y, z) [Gy] from multi-field Linac beam configurations.

    Builds the linear dose influence matrix $A \in \mathbb{R}^{V \times B}$ such that:
    .. math:: \vec{D} = A \vec{w}
    where V is the number of patient voxels and B is the total number of beamlets.
    """

    def __init__(
        self,
        grid_shape: tuple[int, int, int] = (30, 30, 20),
        voxel_size_cm: float = 0.5,
        energy_mv: float = 6.0,
    ) -> None:
        self.shape = grid_shape
        self.voxel_size = float(voxel_size_cm)
        self.kernel = PolyenergeticKernel(energy_mv=energy_mv)

        # Coordinate axes centered at isocenter
        nx, ny, nz = self.shape
        self.x_cm = (np.arange(nx) - (nx - 1) / 2.0) * self.voxel_size
        self.y_cm = (np.arange(ny) - (ny - 1) / 2.0) * self.voxel_size
        self.z_cm = (np.arange(nz) - (nz - 1) / 2.0) * self.voxel_size

        # 3D meshgrid
        self.X, self.Y, self.Z = np.meshgrid(self.x_cm, self.y_cm, self.z_cm, indexing="ij")
        self.n_voxels = nx * ny * nz

        # Relative electron density grid (defaults to water = 1.0)
        self.density = np.ones(self.shape, dtype=np.float64)

    def set_ct_hounsfield_units(self, hu_grid: np.ndarray) -> None:
        """Assign CT Hounsfield Units and compute relative electron density."""
        if hu_grid.shape != self.shape:
            raise ValueError(f"HU grid shape {hu_grid.shape} != engine shape {self.shape}")
        self.density = TissueHeterogeneityCorrector.hu_to_relative_electron_density(hu_grid)

    def calculate_beamlet_dose(
        self,
        beam: LinacBeam,
        x_bev_cm: float,
        y_bev_cm: float,
        beamlet_width_cm: float = 0.5,
    ) -> np.ndarray:
        """Calculate 3D dose distribution deposited by a single beamlet of unit weight [Gy/MU]."""
        # Vector from source to voxel centers
        # Transform all voxel points to BEV
        theta_g = np.radians(beam.gantry_angle)
        cos_g = np.cos(theta_g)
        sin_g = np.sin(theta_g)

        # Distance from central axis along beam direction
        # Ray direction: [-sin_g, -cos_g, 0]
        # Depth along ray from source:
        # Source is at [100 * sin_g, 100 * cos_g, 0]
        src = beam.source_pos
        dx = self.X - src[0]
        dy = self.Y - src[1]
        dz = self.Z - src[2]

        # Depth along beam central axis from surface (approx d = dot(voxel - isocenter, ray_dir) + patient_depth_to_iso)
        ray_dir = beam.central_axis_dir
        depth = (dx * ray_dir[0] + dy * ray_dir[1] + dz * ray_dir[2]) - (beam.sad - 10.0)
        depth = np.maximum(depth, 0.0)

        # Approximate radiological depth scaling by local tissue density
        d_eff = depth * self.density

        # Lateral distance from beamlet ray in BEV plane
        # Voxel position in BEV:
        x_vox_bev = self.X * cos_g - self.Y * sin_g
        y_vox_bev = self.Z

        # Ray divergence scaling at depth: (SAD + depth) / SAD
        div_factor = np.maximum((beam.sad + (depth - 10.0)) / beam.sad, 0.5)
        x_ray = x_bev_cm * div_factor
        y_ray = y_bev_cm * div_factor

        r_bev = np.sqrt((x_vox_bev - x_ray) ** 2 + (y_vox_bev - y_ray) ** 2)

        # PDD and radial scatter
        pdd = self.kernel.percentage_depth_dose(d_eff)
        # Radial pencil Gaussian profiles
        sigma = 0.35 * div_factor  # ~3.5 mm width
        rad_factor = np.exp(-(r_bev ** 2) / (2.0 * (sigma ** 2))) / (2.0 * np.pi * (sigma ** 2))

        # Integrated dose per beamlet
        dose_grid = pdd * rad_factor * (beamlet_width_cm ** 2)
        return dose_grid

    def build_influence_matrix(
        self,
        beams: List[LinacBeam],
        beamlet_coords_per_beam: List[List[Tuple[float, float]]],
    ) -> np.ndarray:
        """Construct full linear dose influence matrix A (V_voxels, B_beamlets)."""
        all_cols = []
        for beam, b_coords in zip(beams, beamlet_coords_per_beam):
            for x_b, y_b in b_coords:
                col_3d = self.calculate_beamlet_dose(beam, x_b, y_b)
                all_cols.append(col_3d.ravel())

        return np.column_stack(all_cols)
