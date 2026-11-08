"""Tests for RadBeam command-line interface."""

import sys
import pytest
from rad_beam.cli import main


def test_cli_help(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rad-beam", "--help"])
    with pytest.raises(SystemExit) as excinfo:
        main()
    assert excinfo.value.code == 0
    captured = capsys.readouterr()
    assert "RadBeam" in captured.out


def test_cli_dose(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rad-beam", "dose", "--gantry", "45.0", "--energy", "6.0"])
    main()
    captured = capsys.readouterr()
    assert "Pencil Beam Dose Profile" in captured.out
    assert "Peak Dose" in captured.out


def test_cli_plan(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["rad-beam", "plan", "--beams", "3", "--dose", "50.0", "--iter", "15"])
    main()
    captured = capsys.readouterr()
    assert "Dose-Volume Histogram (DVH)" in captured.out
    assert "RADIATION ONCOLOGY INVERSE TREATMENT PLAN REPORT" in captured.out
