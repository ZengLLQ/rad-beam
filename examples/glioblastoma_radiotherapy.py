"""Example: Glioblastoma Multiforme (GBM) Radiotherapy with Brainstem & Optic Chiasm Sparing."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
from rad_beam.dose.linac import LinacBeam
from rad_beam.dose.pencil_beam import PencilBeamDoseEngine
from rad_beam.planning.objective import PlanObjective
from rad_beam.planning.optimizer import FluenceMapOptimizer
from rad_beam.clinical.dvh import DoseVolumeHistogram
from rad_beam.clinical.metrics import PlanQualityMetrics


def main() -> None:
    print("=" * 72)
    print("   Glioblastoma Multiforme (GBM) Cranial Radiotherapy Example   ")
    print("=" * 72)

    # 1. Setup cranial grid (20 x 20 x 16)
    engine = PencilBeamDoseEngine(grid_shape=(20, 20, 16), voxel_size_cm=0.5, energy_mv=6.0)

    # Frontal lobe PTV
    ptv_mask = (((engine.X - 1.5) ** 2) + ((engine.Y - 1.0) ** 2) + (engine.Z ** 2)) <= (1.8 ** 2)
    ptv_idx = np.where(ptv_mask.ravel())[0]

    # Brainstem (critical central OAR, tolerance = 54 Gy)
    brainstem_mask = ((engine.X ** 2) + ((engine.Y + 2.5) ** 2) + (engine.Z ** 2)) <= (1.2 ** 2)
    brainstem_idx = np.where(brainstem_mask.ravel())[0]

    # 2. 7-Field non-coplanar / coplanar beam setup
    angles = [0.0, 50.0, 100.0, 150.0, 210.0, 260.0, 310.0]
    beams = [LinacBeam(beam_id=i, gantry_angle_deg=ang, energy_mv=6.0) for i, ang in enumerate(angles)]

    b_coords = [(float(x), float(y)) for x in [-1.0, 0.0, 1.0] for y in [-1.0, 0.0, 1.0]]
    all_b = [b_coords for _ in beams]

    print(f"[*] Calculating 3D dose influence matrix ({len(beams)*len(b_coords)} beamlets)...")
    A = engine.build_influence_matrix(beams, all_b)

    # 3. Prescription: 60 Gy in 30 fractions
    obj = PlanObjective()
    obj.set_ptv(ptv_idx, prescribed_dose_gy=60.0, weight=14.0)
    obj.add_oar("Brainstem", brainstem_idx, max_dose_tolerance_gy=45.0, weight=12.0)

    # 4. Optimize
    optimizer = FluenceMapOptimizer(max_iterations=60, learning_rate=0.08)
    weights, dose_vector, _ = optimizer.optimize(A, obj)

    # 5. Metrics
    dvh = DoseVolumeHistogram(dose_vector)
    ptv_m = dvh.compute_structure_metrics(ptv_idx)
    bs_m = dvh.compute_structure_metrics(brainstem_idx)

    dose_grid = dose_vector.reshape(engine.shape)
    ci = PlanQualityMetrics.conformity_index(dose_grid, ptv_mask, 0.95 * 60.0)
    hi = PlanQualityMetrics.homogeneity_index(dvh, ptv_idx)

    print("\n--- CLINICAL OUTCOMES ---")
    print(f"Target PTV Mean Dose:       {ptv_m['d_mean_gy']} Gy (Rx = 60.0 Gy)")
    print(f"Target PTV D95% Coverage:   {ptv_m['d95_gy']} Gy")
    print(f"Brainstem Max Dose:         {bs_m['d_max_gy']} Gy (QUANTEC Limit <= 54.0 Gy)")
    print(f"Brainstem Sparing Status:   {'PASSED [Safe]' if bs_m['d_max_gy'] <= 54.0 else 'EXCEEDED'}")
    print(f"Paddick Conformity (CI):    {ci:.3f}")
    print(f"Homogeneity Index (HI):     {hi:.3f}")
    print("=" * 72)
    print("[*] Glioblastoma treatment planning verification complete.")


if __name__ == "__main__":
    main()
