from __future__ import annotations

import json

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from converter.cli import main
from converter.onnx_validate import build_onnx_inspection_report


def _save_model(path, nodes, inputs, outputs, initializers=None, opsets=None):
    graph = helper.make_graph(
        nodes,
        "tiny_graph",
        inputs,
        outputs,
        initializer=initializers or [],
    )
    model = helper.make_model(
        graph,
        producer_name="edge-model-inspector-test",
        opset_imports=opsets or [helper.make_operatorsetid("", 13)],
    )
    model.ir_version = 8
    onnx.save(model, path)


def test_onnx_report_static_model_with_inference(tmp_path):
    path = tmp_path / "linear.onnx"
    weight = numpy_helper.from_array(
        np.ones((3, 2), dtype=np.float32), name="weight"
    )
    bias = numpy_helper.from_array(np.zeros((2,), dtype=np.float32), name="bias")
    _save_model(
        path,
        [
            helper.make_node("MatMul", ["input", "weight"], ["hidden"]),
            helper.make_node("Add", ["hidden", "bias"], ["output"]),
        ],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2])],
        [weight, bias],
    )

    report = build_onnx_inspection_report(str(path), input_shape=[1, 3])
    data = report.to_dict()

    assert data["validation"]["checker_passed"] is True
    assert data["validation"]["ort_session_created"] is True
    assert data["validation"]["inference_passed"] is True
    assert data["graph_summary"]["node_count"] == 2
    assert data["graph_summary"]["initializer_count"] == 2
    assert data["parameter_count"] == 8
    assert data["initializer_dtype_histogram"] == {"FLOAT": 2}
    assert {item["op_type"]: item["count"] for item in data["operator_histogram"]} == {
        "Add": 1,
        "MatMul": 1,
    }
    assert data["custom_ops"] == []
    assert any("FP32" in item for item in data["tensorrt_risk_hints"])


def test_validate_onnx_writes_json_and_reports_dynamic_shapes(tmp_path):
    path = tmp_path / "dynamic.onnx"
    report_path = tmp_path / "report.json"
    weight = numpy_helper.from_array(np.ones((3, 2), dtype=np.float32), name="weight")
    _save_model(
        path,
        [helper.make_node("MatMul", ["input", "weight"], ["output"])],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, ["batch", 3])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, ["batch", 2])],
        [weight],
    )

    exit_code = main(
        [
            "validate-onnx",
            str(path),
            "--input-shape",
            "1,3",
            "--json",
            str(report_path),
        ]
    )

    assert exit_code == 0
    data = json.loads(report_path.read_text(encoding="utf-8"))
    assert data["validation"]["inference_passed"] is True
    assert data["inputs"][0]["dynamic_dimensions"] == [{"index": 0, "value": "batch"}]
    assert data["dynamic_shape_warnings"]
    assert any("TensorRT" in item for item in data["tensorrt_risk_hints"])


def test_onnx_report_detects_custom_ops_and_external_data(tmp_path):
    path = tmp_path / "custom.onnx"
    external = helper.make_tensor(
        "external_weight",
        TensorProto.FLOAT,
        [4],
        vals=[],
        raw=False,
    )
    external.data_location = TensorProto.EXTERNAL
    external.external_data.add(key="location", value="external_weight.bin")
    _save_model(
        path,
        [helper.make_node("CustomOp", ["input"], ["output"], domain="com.example")],
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1])],
        [external],
        [
            helper.make_operatorsetid("", 13),
            helper.make_operatorsetid("com.example", 1),
        ],
    )

    report = build_onnx_inspection_report(str(path), input_shape=[1])
    data = report.to_dict()

    assert data["custom_ops"] == [{"domain": "com.example", "op_type": "CustomOp"}]
    assert data["has_external_data"] is True
    assert any("custom" in item.lower() for item in data["tensorrt_risk_hints"])
