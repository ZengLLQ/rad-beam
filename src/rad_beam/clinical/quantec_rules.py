"""QUANTEC and RTOG clinical radiation therapy protocol compliance auditor."""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
from rad_beam.clinical.dvh import DoseVolumeHistogram


class QUANTECAuditor:
    """Audits 3D radiation therapy plans against international clinical guidelines (QUANTEC / RTOG)."""

    STANDARD_CONSTRAINTS = {
        "Rectum": [
            ("V50Gy", 50.0, 50.0, "<="),  # V_50Gy <= 50%
            ("V65Gy", 65.0, 25.0, "<="),  # V_65Gy <= 25%
        ],
        "Bladder": [
            ("V65Gy", 65.0, 50.0, "<="),
            ("V70Gy", 70.0, 35.0, "<="),
        ],
        "SpinalCord": [
            ("Dmax", 45.0, "<="),         # D_max <= 45 Gy
        ],
        "Brainstem": [
            ("Dmax", 54.0, "<="),         # D_max <= 54 Gy
        ],
        "OpticChiasm": [
            ("Dmax", 54.0, "<="),
        ],
    }

    @classmethod
    def audit_plan(
        cls,
        dvh: DoseVolumeHistogram,
        structure_indices: Dict[str, np.ndarray],
        prescribed_dose_gy: float,
    ) -> Dict[str, object]:
        """Perform automated dosimetric audit of target coverage and normal tissue sparing."""
        results: List[Dict[str, object]] = []
        all_passed = True

        # 1. PTV Target Coverage Audit
        if "PTV" in structure_indices:
            ptv_idx = structure_indices["PTV"]
            v95 = dvh.get_vx(ptv_idx, 0.95 * prescribed_dose_gy)
            passed = v95 >= 95.0
            if not passed:
                all_passed = False
            results.append({
                "structure": "PTV",
                "parameter": "V95% Presc",
                "achieved": round(v95, 2),
                "goal": ">= 95.0 %",
                "passed": passed,
            })

        # 2. OAR Sparing Audits
        for struct_name, idx in structure_indices.items():
            if struct_name == "PTV" or len(idx) == 0:
                continue

            metrics = dvh.compute_structure_metrics(idx)
            rules = cls.STANDARD_CONSTRAINTS.get(struct_name, [])

            for rule in rules:
                rule_name = rule[0]
                if rule_name == "Dmax":
                    max_tol = rule[1]
                    achieved = metrics["d_max_gy"]
                    passed = achieved <= max_tol
                    results.append({
                        "structure": struct_name,
                        "parameter": "Dmax (Gy)",
                        "achieved": achieved,
                        "goal": f"<= {max_tol} Gy",
                        "passed": passed,
                    })
                elif rule_name.startswith("V"):
                    dose_val = rule[1]
                    vol_tol = rule[2]
                    achieved = round(dvh.get_vx(idx, dose_val), 2)
                    passed = achieved <= vol_tol
                    results.append({
                        "structure": struct_name,
                        "parameter": f"V{int(dose_val)}Gy (%)",
                        "achieved": achieved,
                        "goal": f"<= {vol_tol} %",
                        "passed": passed,
                    })
                if not passed:
                    all_passed = False

        return {
            "all_passed": all_passed,
            "audit_entries": results,
            "overall_status": "CLINICALLY APPROVED (QUANTEC Compliant)" if all_passed else "REJECTED (Toxicity Violation)",
        }
