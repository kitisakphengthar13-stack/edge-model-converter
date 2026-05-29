from __future__ import annotations

import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from converter.cli import main
from converter.onnx_validate import build_onnx_inspection_report
from converter.readiness import build_readiness_report


def _save_model(path, *, dynamic=False, outputs=None):
    input_shape = ["batch", 3] if dynamic else [1, 3]
    output_defs = outputs or [("output", TensorProto.FLOAT, ["batch", 2] if dynamic else [1, 2])]
    weight = numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), name="weight")
    nodes = [helper.make_node("MatMul", ["input", "weight"], [output_defs[0][0]])]
    graph = helper.make_graph(
        nodes,
        "readiness_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [
            helper.make_tensor_value_info(name, dtype, shape)
            for name, dtype, shape in output_defs
        ],
        initializer=[weight],
    )
    model = helper.make_model(
        graph,
        producer_name="readiness-test",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def _save_multi_output_image_model(path):
    graph = helper.make_graph(
        [
            helper.make_node("Identity", ["input"], ["output0"]),
            helper.make_node("Identity", ["input"], ["output1"]),
        ],
        "segmentation_like_graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 160, 160])],
        [
            helper.make_tensor_value_info("output0", TensorProto.FLOAT, [1, 3, 160, 160]),
            helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 3, 160, 160]),
        ],
    )
    model = helper.make_model(
        graph,
        producer_name="readiness-test",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def test_small_onnx_passes_generic_profile(tmp_path):
    path = tmp_path / "small.onnx"
    _save_model(path)

    report = build_onnx_inspection_report(str(path), input_shape=[1, 3], target="generic")
    readiness = report.to_dict()["readiness"]

    assert readiness["target"] == "generic"
    assert readiness["readiness_level"] == "pass"
    assert readiness["score"] == 100
    assert readiness["findings"] == []


def test_dynamic_shape_triggers_tensorrt_warning(tmp_path):
    path = tmp_path / "dynamic.onnx"
    _save_model(path, dynamic=True)

    report = build_onnx_inspection_report(
        str(path),
        input_shape=[1, 3],
        target="tensorrt-orin-nano",
    )
    findings = report.to_dict()["readiness"]["findings"]

    assert any(item["code"] == "tensorrt_dynamic_shapes" for item in findings)
    assert report.to_dict()["readiness"]["readiness_level"] == "caution"


def test_external_data_huge_memory_and_fp32_rules(tmp_path):
    path = tmp_path / "small.onnx"
    _save_model(path)
    report = build_onnx_inspection_report(str(path), input_shape=[1, 3]).to_dict()
    report["has_external_data"] = True
    report["estimated_initializer_bytes"] = 3 * 1024 * 1024 * 1024
    report["estimated_initializer_size"] = "3.00 GB"
    report["parameter_count"] = 600_000_001

    readiness = build_readiness_report(report, target="tensorrt-orin-nano").to_dict()
    codes = {item["code"]: item["severity"] for item in readiness["findings"]}

    assert codes["external_data"] == "warning"
    assert codes["initializer_memory_over_2gb"] == "error"
    assert codes["params_over_500m"] == "warning"
    assert codes["fp32_only"] == "warning"
    assert readiness["readiness_level"] == "blocked"


def test_segmentation_like_multi_output_triggers_caution(tmp_path):
    path = tmp_path / "seg.onnx"
    _save_multi_output_image_model(path)
    report = build_onnx_inspection_report(str(path), input_shape=[1, 3, 160, 160]).to_dict()
    readiness = build_readiness_report(report, target="onnxruntime-cpu").to_dict()

    assert any(item["code"] == "segmentation_outputs" for item in readiness["findings"])
    assert readiness["readiness_level"] == "caution"


def test_validate_onnx_target_markdown_and_json(tmp_path):
    path = tmp_path / "dynamic.onnx"
    markdown_path = tmp_path / "report.md"
    json_path = tmp_path / "report.json"
    _save_model(path, dynamic=True)

    exit_code = main(
        [
            "validate-onnx",
            str(path),
            "--input-shape",
            "1,3",
            "--target",
            "tensorrt-orin-nano",
            "--format",
            "markdown",
            "--output",
            str(markdown_path),
            "--json",
            str(json_path),
        ]
    )

    assert exit_code == 0
    markdown = markdown_path.read_text(encoding="utf-8")
    assert "## Edge Readiness" in markdown
    assert "tensorrt_dynamic_shapes" in markdown
    assert "`tensorrt_risk`: TensorRT may require explicit min/opt/max" not in markdown
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["readiness"]["target"] == "tensorrt-orin-nano"


def test_fail_on_warning_respects_readiness_warnings(tmp_path):
    path = tmp_path / "small.onnx"
    _save_model(path)

    exit_code = main(
        [
            "validate-onnx",
            str(path),
            "--input-shape",
            "1,3",
            "--target",
            "tensorrt-orin-nano",
            "--fail-on",
            "warning",
        ]
    )

    assert exit_code == 3


def test_scan_summary_includes_readiness(tmp_path):
    path = tmp_path / "small.onnx"
    output = tmp_path / "scan.json"
    _save_model(path)

    exit_code = main(
        [
            "scan",
            str(tmp_path),
            "--include",
            "*.onnx",
            "--input-shape",
            "1,3",
            "--target",
            "tensorrt-orin-nano",
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["artifacts"][0]["report"]["readiness"]["readiness_level"] == "caution"
