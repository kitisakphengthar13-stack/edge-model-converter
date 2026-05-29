from __future__ import annotations

import json

import torch

from converter.inspect_pt import (
    build_inspection_report,
    detect_task_hints,
    framework_from_unsupported_global,
)


def test_report_summarizes_keys_tensors_and_nested_sections(tmp_path):
    path = tmp_path / "model.pt"
    checkpoint = {
        "state_dict": {
            "layer.weight": torch.ones(2, 3, dtype=torch.float32),
            "layer.bias": torch.zeros(2, dtype=torch.float16),
        },
        "hyper_parameters": {"backbone": "tiny"},
        "train_args": {"task": "detect"},
    }
    torch.save(checkpoint, path)

    report = build_inspection_report(path, checkpoint, max_items=10)
    data = report.to_dict()

    assert data["top_level_key_count"] == 3
    assert data["top_level_keys"] == ["state_dict", "hyper_parameters", "train_args"]
    assert data["state_dict_source"] == "state_dict"
    assert data["state_dict_entries"] == 2
    assert data["tensor_summary"]["tensor_count"] == 2
    assert data["tensor_summary"]["parameter_count"] == 8
    assert data["tensor_summary"]["dtype_histogram"]["torch.float32"] == 1
    assert data["tensor_summary"]["dtype_histogram"]["torch.float16"] == 1
    assert {item["key"] for item in data["nested"]} == {
        "state_dict",
        "hyper_parameters",
        "train_args",
    }
    json.dumps(data)


def test_framework_from_safe_load_error_globals():
    ultralytics = framework_from_unsupported_global(
        "Unsupported global: GLOBAL ultralytics.nn.tasks.DetectionModel was not an allowed global"
    )
    anomalib = framework_from_unsupported_global(
        "Unsupported global: GLOBAL anomalib.models.image.patchcore.lightning_model.Patchcore was not an allowed global"
    )

    assert ultralytics is not None
    assert ultralytics.framework == "ultralytics"
    assert ultralytics.confidence == "high"
    assert anomalib is not None
    assert anomalib.framework == "anomalib"
    assert anomalib.confidence == "high"


def test_yolo_task_ranking_ignores_train_metric_false_positive():
    hints = detect_task_hints(
        [
            "train_metrics.metrics/precision(B)",
            "train_metrics.metrics/recall(B)",
            "train_args.nms",
            "train_args.box",
            "train_results.val/box_loss",
        ]
    )

    assert hints
    assert hints[0][0] == "object_detection"
    assert all(item[0] != "embedding_or_metric_learning" for item in hints)


def test_yolo_segmentation_outranks_detection_when_mask_evidence_exists():
    hints = detect_task_hints(
        [
            "train_args.nms",
            "train_args.box",
            "train_results.val/box_loss",
            "train_args.mask_ratio",
            "train_args.retina_masks",
            "train_metrics.val/seg_loss",
            "train_results.val/seg_loss",
        ]
    )

    assert hints
    assert hints[0][0] == "segmentation"
