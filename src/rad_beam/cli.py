"""Command-line interface for RadBeam IMRT/VMAT inverse planning and pencil beam dose engine."""

from __future__ import annotations
import argparse
import sys
import numpy as np

from rad_beam.dose.linac import LinacBeam
from rad_beam.dose.pencil_beam import PencilBeamDoseEngine
from rad_beam.planning.objective import PlanObjective
from rad_beam.planning.optimizer import FluenceMapOptimizer
from rad_beam.planning.mlc_sequencer import SlidingWindowMLCSequencer
from rad_beam.clinical.dvh import DoseVolumeHistogram
from rad_beam.clinical.metrics import PlanQualityMetrics
from rad_beam.clinical.quantec_rules import QUANTECAuditor
from rad_beam.visualizer import RadVisualizer


def run_plan(args: argparse.Namespace) -> None:
    """Run multi-field IMRT inverse planning optimization and clinical audit."""
    print(f"\n[*] Initializing 3D Inverse Treatment Planning ({args.beams}-Field IMRT, Rx={args.dose} Gy)...")

    # 1. Setup 3D patient volume (24 x 24 x 16 voxels, 0.5 cm resolution = 12 x 12 x 8 cm)
    nx, ny, nz = 24, 24, 16
    engine = PencilBeamDoseEngine(grid_shape=(nx, ny, nz), voxel_size_cm=0.5, energy_mv=args.energy)

    # Define spherical PTV centered at isocenter (radius 2.5 cm -> 5 voxels)
    r_ptv = 2.5
    dist_sq = engine.X ** 2 + engine.Y ** 2 + engine.Z ** 2
    ptv_mask = dist_sq <= (r_ptv ** 2)
    ptv_indices = np.where(ptv_mask.ravel())[0]

    # Define adjacent Organ-at-Risk (e.g., Rectum / Spinal Cord shifted 3.5 cm posteriorly)
    oar_mask = ((engine.X) ** 2 + (engine.Y + 3.5) ** 2 + engine.Z ** 2) <= (1.8 ** 2)
    oar_indices = np.where(oar_mask.ravel())[0]

    print(f"[*] Planning Target Volume (PTV): {len(ptv_indices)} voxels ({len(ptv_indices)*0.125:.1f} cm^3)")
    print(f"[*] Organ-at-Risk (OAR):         {len(oar_indices)} voxels ({len(oar_indices)*0.125:.1f} cm^3)")

    # 2. Configure multi-field Linac arrangement (e.g. 5 equispaced gantry angles)
    gantry_angles = np.linspace(0.0, 360.0, args.beams, endpoint=False)
    beams = [LinacBeam(beam_id=i, gantry_angle_deg=float(ang), energy_mv=args.energy) for i, ang in enumerate(gantry_angles)]

    # 5x5 grid of beamlets covering target (-2.0 to +2.0 cm in BEV)
    b_offsets = np.linspace(-2.0, 2.0, 5)
    beamlet_grid = [(float(x), float(y)) for x in b_offsets for y in b_offsets]
    all_beamlets = [beamlet_grid for _ in beams]

    print(f"[*] Calculating 3D Pencil Beam influence matrix ({len(beams)*len(beamlet_grid)} total beamlets)...")
    A = engine.build_influence_matrix(beams, all_beamlets)

    # 3. Formulate inverse optimization objective
    obj = PlanObjective()
    obj.set_ptv(ptv_indices, prescribed_dose_gy=args.dose, weight=12.0)
    obj.add_oar("OAR_Organ", oar_indices, max_dose_tolerance_gy=args.dose * 0.5, weight=8.0)

    # 4. Solve constrained fluence map optimization via Projected Gradient Descent
    print(f"[*] Running Fluence Map Optimization (PGD, {args.iter} iterations)...")
    optimizer = FluenceMapOptimizer(max_iterations=args.iter, learning_rate=0.08)
    weights, dose_vector, cost_history = optimizer.optimize(A, obj)

    dose_grid_3d = dose_vector.reshape(engine.shape)

    # 5. Dynamic MLC Leaf Sequencing
    sequencer = SlidingWindowMLCSequencer(leaf_width_cm=0.5)
    # Take fluence from first beam (5x5 matrix)
    f_beam0 = weights[:25].reshape(5, 5)
    control_points = sequencer.sequence_fluence_map(f_beam0, num_control_points=8)
    print(f"[*] Decomposed Beam 1 fluence map into {len(control_points)} dynamic MLC control points.")

    # 6. Clinical DVH and Plan Metrics
    dvh = DoseVolumeHistogram(dose_vector, num_bins=120)
    ptv_dvh = dvh.cumulative_dvh(ptv_indices)
    oar_dvh = dvh.cumulative_dvh(oar_indices)

    ptv_metrics = dvh.compute_structure_metrics(ptv_indices)
    ptv_metrics["d_prescribed_gy"] = args.dose
    oar_metrics = {"OAR_Organ": dvh.compute_structure_metrics(oar_indices)}

    hi = PlanQualityMetrics.homogeneity_index(dvh, ptv_indices)
    ci = PlanQualityMetrics.conformity_index(dose_grid_3d, ptv_mask, reference_isodose_gy=0.95 * args.dose)

    audit = QUANTECAuditor.audit_plan(dvh, {"PTV": ptv_indices, "Rectum": oar_indices}, args.dose)

    # 7. Render visual ASCII reports
    print("\n" + RadVisualizer.render_dvh_ascii(ptv_dvh, oar_dvh, oar_name="OAR"))
    print("\n" + RadVisualizer.format_plan_report(ptv_metrics, oar_metrics, hi, ci, audit))
    print("")


def run_dose(args: argparse.Namespace) -> None:
    """Calculate single beam 3D pencil beam dose distribution."""
    engine = PencilBeamDoseEngine(grid_shape=(20, 20, 20), voxel_size_cm=0.5, energy_mv=args.energy)
    beam = LinacBeam(beam_id=0, gantry_angle_deg=args.gantry, energy_mv=args.energy)
    dose_3d = engine.calculate_beamlet_dose(beam, x_bev_cm=0.0, y_bev_cm=0.0)

    print(f"\n[*] 3D Pencil Beam Dose Profile (Gantry={args.gantry} deg, E={args.energy} MV)")
    print(f"    - Peak Dose:            {np.max(dose_3d):.4f} Gy/MU")
    print(f"    - Mean Volume Dose:     {np.mean(dose_3d):.4f} Gy/MU")
    print(f"    - Maximum Depth Dose:   d_max = {engine.kernel.d_max} cm\n")


def main() -> None:
    """CLI entrypoint."""
    parser = argparse.ArgumentParser(
        prog="rad-beam",
        description="RadBeam: 3D Megavoltage Pencil Beam Convolution & IMRT/VMAT Inverse Planning Engine",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Plan subcommand
    plan_p = subparsers.add_parser("plan", help="Run multi-field IMRT inverse planning")
    plan_p.add_argument("--beams", type=int, default=5, help="Number of equispaced gantry angles (default: 5)")
    plan_p.add_argument("--dose", type=float, default=60.0, help="Prescribed target dose in Gy (default: 60.0)")
    plan_p.add_argument("--energy", type=float, default=6.0, help="Photon beam energy in MV (default: 6.0)")
    plan_p.add_argument("--iter", type=int, default=80, help="Max optimization iterations (default: 80)")

    # Dose subcommand
    dose_p = subparsers.add_parser("dose", help="Calculate single beam pencil beam dose distribution")
    dose_p.add_argument("--gantry", type=float, default=0.0, help="Gantry angle in degrees (default: 0.0)")
    dose_p.add_argument("--energy", type=float, default=6.0, help="Photon beam energy in MV (default: 6.0)")

    args = parser.parse_args()
    if args.command == "plan":
        run_plan(args)
    elif args.command == "dose":
        run_dose(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
