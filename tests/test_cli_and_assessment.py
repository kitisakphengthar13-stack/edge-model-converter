from __future__ import annotations

import json

import torch

from converter.cli import main
from converter.export_assessment import assess_export_from_model_path


def test_inspect_writes_json_report(tmp_path):
    checkpoint_path = tmp_path / "tiny.pt"
    report_path = tmp_path / "report.json"
    torch.save({"state_dict": {"w": torch.ones(1)}}, checkpoint_path)

    exit_code = main(
        [
            "inspect",
            str(checkpoint_path),
            "--json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["checkpoint_kind"] == "checkpoint dict"
    assert data["state_dict_entries"] == 1
    assert data["tensor_summary"]["parameter_count"] == 1


def test_assess_export_uses_safe_load_framework_error_for_route(tmp_path, monkeypatch):
    checkpoint_path = tmp_path / "best.pt"
    checkpoint_path.write_bytes(b"not actually loaded")

    def fail_safe(_path):
        raise RuntimeError(
            "Weights only load failed. Unsupported global: GLOBAL "
            "ultralytics.nn.tasks.SegmentationModel was not an allowed global"
        )

    monkeypatch.setattr("converter.export_assessment.load_checkpoint_safe", fail_safe)
    result = assess_export_from_model_path(str(checkpoint_path), unsafe_load=False)

    assert result["detected_source"]["framework"] == "ultralytics"
    assert result["detected_source"]["model_family"] == "yolo"
    assert result["detected_source"]["task"] == "segmentation"
    assert result["recommended_route"]["route"] == "official_source_exporter"
