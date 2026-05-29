from __future__ import annotations

from pathlib import Path
from typing import Any

from .onnx_validate import build_onnx_inspection_report
from .reports import ReportMessage


def compare_onnx(left: str, right: str) -> dict[str, Any]:
    left_report = build_onnx_inspection_report(left).to_dict()
    right_report = build_onnx_inspection_report(right).to_dict()
    differences: list[ReportMessage] = []

    def add(name: str, left_value: Any, right_value: Any, severity: str = "warning") -> None:
        if left_value != right_value:
            differences.append(
                ReportMessage(
                    severity,
                    f"{name} changed: left={left_value!r}, right={right_value!r}",
                    name,
                )
            )

    add("size_bytes", left_report["size_bytes"], right_report["size_bytes"], "info")
    add("inputs", _io_signature(left_report["inputs"]), _io_signature(right_report["inputs"]))
    add("outputs", _io_signature(left_report["outputs"]), _io_signature(right_report["outputs"]))
    add(
        "node_count",
        left_report["graph_summary"]["node_count"],
        right_report["graph_summary"]["node_count"],
    )
    add("parameter_count", left_report["parameter_count"], right_report["parameter_count"])
    add(
        "initializer_dtype_histogram",
        left_report["initializer_dtype_histogram"],
        right_report["initializer_dtype_histogram"],
    )
    add("operator_histogram", _op_map(left_report), _op_map(right_report))
    add(
        "dynamic_shape_warnings",
        bool(left_report["dynamic_shape_warnings"]),
        bool(right_report["dynamic_shape_warnings"]),
    )
    add(
        "warning_count",
        len(left_report["dynamic_shape_warnings"]) + len(left_report["tensorrt_risk_hints"]),
        len(right_report["dynamic_shape_warnings"]) + len(right_report["tensorrt_risk_hints"]),
        "info",
    )

    return {
        "left": left_report,
        "right": right_report,
        "differences": [item.__dict__ for item in differences],
        "same": not differences,
    }


def render_compare_text(result: dict[str, Any]) -> str:
    left = result["left"]
    right = result["right"]
    lines = [
        "ONNX comparison",
        "---------------",
        f"Left: {left['path']}",
        f"Right: {right['path']}",
        f"Same: {result['same']}",
        "",
        "Summary",
        "-------",
        f"Size bytes: {left['size_bytes']} -> {right['size_bytes']}",
        f"Node count: {left['graph_summary']['node_count']} -> {right['graph_summary']['node_count']}",
        f"Parameter count: {left['parameter_count']} -> {right['parameter_count']}",
        f"Inputs: {_io_signature(left['inputs'])} -> {_io_signature(right['inputs'])}",
        f"Outputs: {_io_signature(left['outputs'])} -> {_io_signature(right['outputs'])}",
        f"Ops: {_op_map(left)} -> {_op_map(right)}",
        "",
        "Differences",
        "-----------",
    ]
    if result["differences"]:
        lines.extend(
            f"- {item['severity']} {item.get('code') or ''}: {item['message']}"
            for item in result["differences"]
        )
    else:
        lines.append("None")
    return "\n".join(lines) + "\n"


def _io_signature(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "name": item["name"],
            "shape": item["shape"],
            "dtype": item["dtype"],
            "dynamic": bool(item["dynamic_dimensions"]),
        }
        for item in values
    ]


def _op_map(report: dict[str, Any]) -> dict[str, int]:
    return {
        f"{item['domain']}::{item['op_type']}": item["count"]
        for item in report["operator_histogram"]
    }
