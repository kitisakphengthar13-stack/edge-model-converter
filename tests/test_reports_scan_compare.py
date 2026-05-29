from __future__ import annotations

import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from converter.cli import main


def _save_linear(path, *, out_features=2, dynamic=False):
    input_shape = ["batch", 3] if dynamic else [1, 3]
    output_shape = ["batch", out_features] if dynamic else [1, out_features]
    weight = numpy_helper.from_array(
        np.ones((3, out_features), dtype=np.float32), name="weight"
    )
    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        "graph",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
        initializer=[weight],
    )
    model = helper.make_model(
        graph,
        producer_name="test",
        opset_imports=[helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def test_markdown_output_for_validate_onnx(tmp_path):
    model_path = tmp_path / "model.onnx"
    markdown_path = tmp_path / "report.md"
    _save_linear(model_path)

    exit_code = main(
        [
            "validate-onnx",
            str(model_path),
            "--input-shape",
            "1,3",
            "--format",
            "markdown",
            "--output",
            str(markdown_path),
        ]
    )

    assert exit_code == 0
    text = markdown_path.read_text(encoding="utf-8")
    assert "# Inspection Report" in text
    assert "## ONNX Graph" in text
    assert "ai.onnx::MatMul" in text


def test_scan_mixed_valid_invalid_outputs_json_and_markdown(tmp_path):
    valid_path = tmp_path / "valid.onnx"
    invalid_path = tmp_path / "invalid.onnx"
    json_path = tmp_path / "scan.json"
    md_path = tmp_path / "scan.md"
    _save_linear(valid_path)
    invalid_path.write_text("not onnx", encoding="utf-8")

    exit_code = main(
        [
            "scan",
            str(tmp_path),
            "--include",
            "*.onnx",
            "--input-shape",
            "1,3",
            "--format",
            "json",
            "--output",
            str(json_path),
        ]
    )
    assert exit_code == 0
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert len(data["artifacts"]) == 2
    assert any(item["inspection_status"] == "error" for item in data["artifacts"])
    assert any(item["inspection_status"] == "ok" for item in data["artifacts"])

    exit_code = main(
        [
            "scan",
            str(tmp_path),
            "--include",
            "*.onnx",
            "--input-shape",
            "1,3",
            "--format",
            "markdown",
            "--output",
            str(md_path),
        ]
    )
    assert exit_code == 0
    assert "| File | Kind | Status |" in md_path.read_text(encoding="utf-8")


def test_compare_onnx_json(tmp_path):
    left = tmp_path / "left.onnx"
    right = tmp_path / "right.onnx"
    output = tmp_path / "compare.json"
    _save_linear(left, out_features=2)
    _save_linear(right, out_features=4)

    exit_code = main(
        [
            "compare",
            str(left),
            str(right),
            "--format",
            "json",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    data = json.loads(output.read_text(encoding="utf-8"))
    assert data["same"] is False
    assert any(item["code"] == "outputs" for item in data["differences"])
    assert any(item["code"] == "parameter_count" for item in data["differences"])


def test_fail_on_warning_and_json_backward_compatibility(tmp_path):
    model_path = tmp_path / "dynamic.onnx"
    json_path = tmp_path / "legacy.json"
    _save_linear(model_path, dynamic=True)

    exit_code = main(
        [
            "validate-onnx",
            str(model_path),
            "--input-shape",
            "1,3",
            "--json",
            str(json_path),
            "--fail-on",
            "warning",
        ]
    )

    assert exit_code == 3
    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert "dynamic_shape_warnings" in data
    assert "report" not in data
