"""Terminal and ASCII visualizer for Dose-Volume Histograms (DVH) and isodose distributions."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np


class RadVisualizer:
    """Renders formatted ASCII DVH plots and clinical radiation oncology reports."""

    @staticmethod
    def render_dvh_ascii(
        dvh_ptv: tuple[np.ndarray, np.ndarray],
        dvh_oar: tuple[np.ndarray, np.ndarray] | None = None,
        oar_name: str = "OAR",
        width: int = 50,
        height: int = 10,
    ) -> str:
        """Render high-clarity dual-structure cumulative DVH curve in ASCII."""
        d_bins_p, v_ptv = dvh_ptv
        max_d = float(np.max(d_bins_p))

        # Grid of height rows (100% down to 0%) and width columns (0 to max_d Gy)
        grid = [[" " for _ in range(width)] for _ in range(height)]

        # Sample PTV
        sample_indices = np.linspace(0, len(d_bins_p) - 1, width, dtype=int)
        for col, s_idx in enumerate(sample_indices):
            v_val = v_ptv[s_idx]  # 0 to 100
            row = int((height - 1) * (1.0 - (v_val / 100.0)))
            row = min(max(row, 0), height - 1)
            grid[row][col] = "P"  # PTV marker

        # Sample OAR if provided
        if dvh_oar is not None:
            d_bins_o, v_oar = dvh_oar
            for col, s_idx in enumerate(sample_indices):
                v_val = v_oar[s_idx]
                row = int((height - 1) * (1.0 - (v_val / 100.0)))
                row = min(max(row, 0), height - 1)
                if grid[row][col] == " ":
                    grid[row][col] = "O"  # OAR marker
                elif grid[row][col] == "P":
                    grid[row][col] = "#"  # Overlap marker

        lines = [
            "=== Dose-Volume Histogram (DVH) [P = PTV, O = " + oar_name + "] ===",
            "  Vol (%)",
        ]
        for r in range(height):
            vol_level = 100.0 * (1.0 - r / (height - 1))
            row_str = "".join(grid[r])
            lines.append(f" {vol_level:5.0f} % | {row_str}")

        lines.append(f"        +{'-' * width}")
        lines.append(f"        0.0 Gy{' ' * (width - 14)}{max_d:.1f} Gy (Dose)")
        return "\n".join(lines)

    @staticmethod
    def format_plan_report(
        ptv_metrics: Dict[str, float],
        oar_metrics: Dict[str, Dict[str, float]],
        homogeneity_idx: float,
        conformity_idx: float,
        audit: Dict[str, object],
    ) -> str:
        """Format an elegant clinical radiation therapy plan evaluation report."""
        lines = [
            "==================================================================",
            "         RADIATION ONCOLOGY INVERSE TREATMENT PLAN REPORT         ",
            "==================================================================",
            " 1. TARGET DOSIMETRY (PTV)",
            f"    - Prescribed Target Dose:   {ptv_metrics.get('d_prescribed_gy', '60.0')} Gy",
            f"    - Target D95% Coverage:     {ptv_metrics.get('d95_gy', 0.0)} Gy",
            f"    - Target D98% (Near-Min):   {ptv_metrics.get('d98_gy', 0.0)} Gy",
            f"    - Target D2%  (Near-Max):   {ptv_metrics.get('d2_gy', 0.0)} Gy",
            f"    - Mean Target Dose:         {ptv_metrics.get('d_mean_gy', 0.0)} Gy",
            f"    - Homogeneity Index (HI):   {homogeneity_idx:.3f}",
            f"    - Conformity Index (CI):    {conformity_idx:.3f}",
            "",
            " 2. ORGANS-AT-RISK (OAR) SPARING",
        ]
        for oar_name, m in oar_metrics.items():
            lines.append(f"    - [{oar_name}] Mean: {m['d_mean_gy']} Gy, Max: {m['d_max_gy']} Gy, D50%: {m['d50_gy']} Gy")

        lines.extend([
            "",
            " 3. QUANTEC / RTOG CLINICAL AUDIT",
            f"    Overall Plan Status: {audit['overall_status']}",
            "    --------------------------------------------------------------",
            f"    {'STRUCTURE':<14} | {'PARAMETER':<16} | {'ACHIEVED':<10} | {'GOAL':<12} | {'RESULT'}",
            "    --------------------------------------------------------------",
        ])
        for entry in audit["audit_entries"]:
            status_str = "PASS" if entry["passed"] else "FAIL"
            lines.append(
                f"    {entry['structure']:<14} | {entry['parameter']:<16} | "
                f"{entry['achieved']:<10} | {entry['goal']:<12} | [{status_str}]"
            )
        lines.append("==================================================================")
        return "\n".join(lines)
