"""Unit tests for QUANTEC auditing and plan quality metrics."""

import pytest
import numpy as np
from rad_beam.clinical.dvh import DoseVolumeHistogram
from rad_beam.clinical.metrics import PlanQualityMetrics
from rad_beam.clinical.quantec_rules import QUANTECAuditor


def test_quantec_compliance():
    # 200 voxels: 0..99 is PTV (dose = 60 Gy), 100..199 is Rectum (dose = 30 Gy)
    doses = np.zeros(200)
    doses[:100] = 60.0  # PTV
    doses[100:] = 30.0  # Rectum (well below 50 Gy)

    dvh = DoseVolumeHistogram(doses)
    ptv_idx = np.arange(100)
    rectum_idx = np.arange(100, 200)

    audit = QUANTECAuditor.audit_plan(
        dvh,
        {"PTV": ptv_idx, "Rectum": rectum_idx},
        prescribed_dose_gy=60.0,
    )
    assert audit["all_passed"] is True
    assert "CLINICALLY APPROVED" in audit["overall_status"]


def test_plan_quality_metrics():
    doses = np.ones(100) * 60.0
    dvh = DoseVolumeHistogram(doses)
    ptv_idx = np.arange(100)

    # Perfect homogeneity: D2% == D98% -> HI = 0.0
    hi = PlanQualityMetrics.homogeneity_index(dvh, ptv_idx)
    assert hi == pytest.approx(0.0, abs=1e-3)

    # 3D grid
    grid = np.zeros((10, 10, 10))
    ptv_mask = np.zeros((10, 10, 10), dtype=bool)
    ptv_mask[3:7, 3:7, 3:7] = True
    grid[ptv_mask] = 60.0

    ci = PlanQualityMetrics.conformity_index(grid, ptv_mask, reference_isodose_gy=57.0)
    # Perfect match between isodose volume and PTV volume
    assert ci == pytest.approx(1.0)
