"""Example: 5-Field IMRT Prostate Cancer Treatment Planning with Rectum & Bladder Sparing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from rad_beam.dose.linac import LinacBeam
from rad_beam.dose.pencil_beam import PencilBeamDoseEngine
from rad_beam.planning.objective import PlanObjective
from rad_beam.planning.optimizer import FluenceMapOptimizer
from rad_beam.clinical.dvh import DoseVolumeHistogram
from rad_beam.clinical.quantec_rules import QUANTECAuditor


def main() -> None:
    print("=" * 68)
    print("   5-Field IMRT Prostate Carcinoma Inverse Planning Example   ")
    print("=" * 68)

    # 1. Setup 3D pelvic volume (20 x 20 x 14 voxels)
    engine = PencilBeamDoseEngine(grid_shape=(20, 20, 14), voxel_size_cm=0.5, energy_mv=6.0)

    # Prostate PTV (radius 2.0 cm)
    dist_sq = engine.X ** 2 + engine.Y ** 2 + engine.Z ** 2
    ptv_mask = dist_sq <= (2.0 ** 2)
    ptv_idx = np.where(ptv_mask.ravel())[0]

    # Rectum (posterior organ, shifted y = -3.0 cm)
    rectum_mask = ((engine.X) ** 2 + (engine.Y + 3.0) ** 2 + engine.Z ** 2) <= (1.5 ** 2)
    rectum_idx = np.where(rectum_mask.ravel())[0]

    # Bladder (anterior organ, shifted y = +3.0 cm)
    bladder_mask = ((engine.X) ** 2 + (engine.Y - 3.0) ** 2 + engine.Z ** 2) <= (1.8 ** 2)
    bladder_idx = np.where(bladder_mask.ravel())[0]

    # 2. 5-Field IMRT gantry angles (0, 72, 144, 216, 288 deg)
    angles = [0.0, 72.0, 144.0, 216.0, 288.0]
    beams = [LinacBeam(beam_id=i, gantry_angle_deg=ang, energy_mv=6.0) for i, ang in enumerate(angles)]

    # Beamlet grid 4x4
    b_coords = [(float(x), float(y)) for x in [-1.5, -0.5, 0.5, 1.5] for y in [-1.5, -0.5, 0.5, 1.5]]
    all_b = [b_coords for _ in beams]

    print(f"[*] Building 3D Pencil Beam influence matrix ({len(beams)*len(b_coords)} beamlets)...")
    A = engine.build_influence_matrix(beams, all_b)

    # 3. Objective formulation (Prescription = 78 Gy)
    obj = PlanObjective()
    obj.set_ptv(ptv_idx, prescribed_dose_gy=78.0, weight=15.0)
    obj.add_oar("Rectum", rectum_idx, max_dose_tolerance_gy=45.0, weight=8.0)
    obj.add_oar("Bladder", bladder_idx, max_dose_tolerance_gy=55.0, weight=6.0)

    # 4. Optimize
    optimizer = FluenceMapOptimizer(max_iterations=60, learning_rate=0.08)
    weights, dose_vector, _ = optimizer.optimize(A, obj)

    # 5. Clinical DVH evaluation
    dvh = DoseVolumeHistogram(dose_vector)
    ptv_metrics = dvh.compute_structure_metrics(ptv_idx)
    rectum_metrics = dvh.compute_structure_metrics(rectum_idx)

    print("\n--- RESULTS ---")
    print(f"Prescription Dose:       78.0 Gy")
    print(f"PTV Mean Dose:           {ptv_metrics['d_mean_gy']} Gy")
    print(f"PTV D95% Coverage:       {ptv_metrics['d95_gy']} Gy")
    print(f"Rectum Mean Dose:        {rectum_metrics['d_mean_gy']} Gy")
    print(f"Rectum Max Dose:         {rectum_metrics['d_max_gy']} Gy")

    audit = QUANTECAuditor.audit_plan(dvh, {"PTV": ptv_idx, "Rectum": rectum_idx}, 78.0)
    print(f"\nQUANTEC Audit Status:    {audit['overall_status']}")
    print("=" * 68)


if __name__ == "__main__":
    main()
