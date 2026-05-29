from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .readiness import build_readiness_report
from .spec import validate_spec_file
from .utils import format_file_size, print_section

DTYPE_TO_NUMPY = {
    "float32": "float32",
    "float16": "float16",
    "int64": "int64",
    "int32": "int32",
    "uint8": "uint8",
    "bool": "bool",
}

ONNX_TYPE_TO_DTYPE = {
    "tensor(float)": "float32",
    "tensor(float16)": "float16",
    "tensor(int64)": "int64",
    "tensor(int32)": "int32",
    "tensor(uint8)": "uint8",
    "tensor(bool)": "bool",
}

STANDARD_ONNX_DOMAINS = {"", "ai.onnx"}
LARGE_INITIALIZER_BYTES = 100 * 1024 * 1024


@dataclass
class OnnxValueReport:
    name: str
    shape: list[Any]
    dtype: str
    dynamic_dimensions: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class OnnxInitializerReport:
    name: str
    shape: list[int]
    dtype: str
    parameter_count: int
    estimated_bytes: int
    estimated_size: str
    external_data: bool = False


@dataclass
class OnnxValidationStatus:
    checker_passed: bool = False
    checker_error: str | None = None
    ort_session_created: bool = False
    ort_error: str | None = None
    inference_attempted: bool = False
    inference_passed: bool = False
    inference_error: str | None = None
    inference_skipped_reason: str | None = None


@dataclass
class OnnxInspectionReport:
    path: str
    size_bytes: int
    size: str
    ir_version: int
    producer_name: str
    producer_version: str
    graph_name: str
    opset_imports: list[dict[str, Any]]
    inputs: list[OnnxValueReport]
    outputs: list[OnnxValueReport]
    graph_summary: dict[str, int]
    parameter_count: int
    estimated_initializer_bytes: int
    estimated_initializer_size: str
    initializer_dtype_histogram: dict[str, int]
    largest_initializers: list[OnnxInitializerReport]
    operator_histogram: list[dict[str, Any]]
    custom_ops: list[dict[str, str]]
    has_external_data: bool
    dynamic_shape_warnings: list[str]
    tensorrt_risk_hints: list[str]
    validation: OnnxValidationStatus
    output_summaries: list[dict[str, Any]] = field(default_factory=list)
    readiness: dict[str, Any] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_onnx_file(
    onnx_path: str,
    spec_path: str | None = None,
    input_shape: list[int] | None = None,
    input_dtype: str | None = None,
    input_name: str | None = None,
    max_items: int = 20,
    target: str = "generic",
) -> dict[str, Any]:
    report = build_onnx_inspection_report(
        onnx_path,
        spec_path=spec_path,
        input_shape=input_shape,
        input_dtype=input_dtype,
        input_name=input_name,
        max_items=max_items,
        target=target,
    )
    print_onnx_report(report, max_items=max_items)
    return report.to_dict() | {
        "onnx_path": report.path,
        "checker_passed": report.validation.checker_passed,
        "ort_session_created": report.validation.ort_session_created,
        "inference_passed": report.validation.inference_passed,
        "input_names": [item.name for item in report.inputs],
        "output_names": [item.name for item in report.outputs],
        "success": report.validation.checker_passed and report.validation.ort_session_created,
    }


def build_onnx_inspection_report(
    onnx_path: str,
    spec_path: str | None = None,
    input_shape: list[int] | None = None,
    input_dtype: str | None = None,
    input_name: str | None = None,
    max_items: int = 20,
    target: str = "generic",
) -> OnnxInspectionReport:
    path = Path(onnx_path)
    if not path.is_file():
        raise RuntimeError(f"ONNX file not found: {path}")

    try:
        import numpy as np
        import onnx
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        missing = exc.name or "required ONNX validation package"
        raise RuntimeError(
            f"Missing ONNX validation dependency '{missing}'. Install requirements.txt."
        ) from exc

    model = onnx.load(path, load_external_data=False)
    inputs = [_value_report(value) for value in model.graph.input]
    outputs = [_value_report(value) for value in model.graph.output]
    initializers = [_initializer_report(onnx, item) for item in model.graph.initializer]
    operator_histogram = _operator_histogram(model)
    custom_ops = _custom_ops(model)
    validation = OnnxValidationStatus()

    try:
        onnx.checker.check_model(str(path))
        validation.checker_passed = True
    except Exception as exc:
        validation.checker_error = str(exc)

    session = None
    ort_inputs: list[Any] = []
    ort_outputs: list[Any] = []
    try:
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        validation.ort_session_created = True
        ort_inputs = list(session.get_inputs())
        ort_outputs = list(session.get_outputs())
    except Exception as exc:
        validation.ort_error = str(exc)

    output_summaries: list[dict[str, Any]] = []
    if session is not None and len(ort_inputs) == 1:
        try:
            resolved_name, resolved_shape, resolved_dtype = _resolve_input_request(
                ort_inputs[0],
                spec_path=spec_path,
                input_shape=input_shape,
                input_dtype=input_dtype,
                input_name=input_name,
            )
            dummy_input = _create_numpy_dummy_input(np, resolved_shape, resolved_dtype)
            validation.inference_attempted = True
            ort_values = session.run(None, {resolved_name: dummy_input})
            validation.inference_passed = True
            output_summaries = _summarize_outputs(ort_outputs, ort_values, max_items=max_items)
        except Exception as exc:
            if input_shape is None and _looks_like_shape_resolution_error(str(exc)):
                validation.inference_skipped_reason = str(exc)
            else:
                validation.inference_attempted = True
                validation.inference_error = str(exc)
    elif session is not None:
        validation.inference_skipped_reason = (
            "single-input dummy inference is currently supported; "
            f"model has {len(ort_inputs)} inputs"
        )

    dynamic_warnings = _dynamic_shape_warnings(inputs + outputs)
    initializer_dtype_histogram: dict[str, int] = {}
    for item in initializers:
        initializer_dtype_histogram[item.dtype] = initializer_dtype_histogram.get(item.dtype, 0) + 1

    estimated_bytes = sum(item.estimated_bytes for item in initializers)
    parameter_count = sum(item.parameter_count for item in initializers)
    largest_initializers = sorted(
        initializers, key=lambda item: item.estimated_bytes, reverse=True
    )[:max_items]

    report = OnnxInspectionReport(
        path=str(path),
        size_bytes=path.stat().st_size,
        size=format_file_size(path.stat().st_size),
        ir_version=model.ir_version,
        producer_name=model.producer_name or "-",
        producer_version=model.producer_version or "-",
        graph_name=model.graph.name or "-",
        opset_imports=[
            {"domain": item.domain or "ai.onnx", "version": item.version}
            for item in model.opset_import
        ],
        inputs=inputs,
        outputs=outputs,
        graph_summary={
            "node_count": len(model.graph.node),
            "initializer_count": len(model.graph.initializer),
            "value_info_count": len(model.graph.value_info),
            "input_count": len(model.graph.input),
            "output_count": len(model.graph.output),
        },
        parameter_count=parameter_count,
        estimated_initializer_bytes=estimated_bytes,
        estimated_initializer_size=format_file_size(estimated_bytes),
        initializer_dtype_histogram=initializer_dtype_histogram,
        largest_initializers=largest_initializers,
        operator_histogram=operator_histogram,
        custom_ops=custom_ops,
        has_external_data=any(item.external_data for item in initializers),
        dynamic_shape_warnings=dynamic_warnings,
        tensorrt_risk_hints=_tensorrt_risk_hints(
            dynamic_warnings=dynamic_warnings,
            custom_ops=custom_ops,
            initializers=initializers,
            dtype_histogram=initializer_dtype_histogram,
        ),
        validation=validation,
        output_summaries=output_summaries,
    )
    report.readiness = build_readiness_report(report.to_dict(), target=target).to_dict()
    return report


def parse_cli_shape(value: str) -> list[int]:
    try:
        shape = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise RuntimeError("--input-shape must contain comma-separated integers") from exc
    if not shape:
        raise RuntimeError("--input-shape must not be empty")
    return shape


def print_onnx_report(report: OnnxInspectionReport, max_items: int = 20) -> None:
    print_section("ONNX validation")
    print(f"Path: {report.path}")

    print_section("ONNX model")
    print(f"Size: {report.size}")
    print(f"IR version: {report.ir_version}")
    print(f"Producer name: {report.producer_name}")
    print(f"Producer version: {report.producer_version}")
    print(f"Graph name: {report.graph_name}")
    print(
        "Opset imports: "
        + ", ".join(f"{item['domain']}:{item['version']}" for item in report.opset_imports)
    )

    print_section("Graph summary")
    for key, value in report.graph_summary.items():
        print(f"{key}: {value}")
    print(f"Parameter count: {report.parameter_count}")
    print(f"Estimated initializer memory: {report.estimated_initializer_size}")
    print("Initializer dtype histogram:")
    if report.initializer_dtype_histogram:
        for dtype, count in sorted(report.initializer_dtype_histogram.items()):
            print(f"- {dtype}: {count}")
    else:
        print("- None")

    _print_value_reports("Model inputs", report.inputs, max_items=max_items)
    _print_value_reports("Model outputs", report.outputs, max_items=max_items)

    print_section("Operator histogram")
    for item in report.operator_histogram[:max_items]:
        print(f"- {item['domain']}::{item['op_type']}: {item['count']}")
    if len(report.operator_histogram) > max_items:
        print(f"... {len(report.operator_histogram) - max_items} more")

    print_section("Custom/non-standard ops")
    if report.custom_ops:
        for item in report.custom_ops[:max_items]:
            print(f"- {item['domain']}::{item['op_type']}")
    else:
        print("None")

    print_section("Largest initializers")
    if report.largest_initializers:
        for item in report.largest_initializers[:max_items]:
            print(
                f"- {item.name}: shape={item.shape}, dtype={item.dtype}, "
                f"params={item.parameter_count}, estimated_size={item.estimated_size}, "
                f"external_data={'yes' if item.external_data else 'no'}"
            )
    else:
        print("None")

    print_section("Warnings")
    warnings = report.dynamic_shape_warnings + report.tensorrt_risk_hints
    if warnings:
        for warning in warnings:
            print(f"- {warning}")
    else:
        print("None")

    print_section("Validation status")
    print(f"ONNX checker: {'passed' if report.validation.checker_passed else 'failed'}")
    if report.validation.checker_error:
        print(f"Checker error: {report.validation.checker_error}")
    print(f"ONNX Runtime session: {'created' if report.validation.ort_session_created else 'failed'}")
    if report.validation.ort_error:
        print(f"ORT error: {report.validation.ort_error}")
    if report.validation.inference_passed:
        print("ONNX Runtime inference: passed")
    elif report.validation.inference_error:
        print(f"ONNX Runtime inference: failed: {report.validation.inference_error}")
    elif report.validation.inference_skipped_reason:
        print(f"ONNX Runtime inference: skipped: {report.validation.inference_skipped_reason}")
    else:
        print("ONNX Runtime inference: not attempted")

    if report.output_summaries:
        print_section("Output summary")
        for item in report.output_summaries[:max_items]:
            print(f"- {item['name']}: shape={item['shape']}, dtype={item['dtype']}")
            if "min" in item:
                print(
                    f"  min={item['min']:.6g}, max={item['max']:.6g}, "
                    f"mean={item['mean']:.6g}"
                )

    if report.readiness:
        print_section("Edge readiness")
        print(f"Target: {report.readiness['target']}")
        print(f"Readiness level: {report.readiness['readiness_level']}")
        print(f"Score: {report.readiness['score']}")
        findings = report.readiness.get("findings", [])
        if findings:
            print("Findings:")
            for item in findings[:max_items]:
                print(f"- {item['severity']} [{item['code']}]: {item['message']}")
        else:
            print("Findings: none")
        actions = report.readiness.get("recommended_next_actions", [])
        if actions:
            print("Recommended next actions:")
            for item in actions[:max_items]:
                print(f"- {item}")


def _print_value_reports(title: str, values: list[OnnxValueReport], max_items: int) -> None:
    print_section(title)
    for value in values[:max_items]:
        print(f"- name={value.name}, shape={value.shape}, dtype={value.dtype}")
        if value.dynamic_dimensions:
            print(f"  dynamic_dimensions={value.dynamic_dimensions}")
    if len(values) > max_items:
        print(f"... {len(values) - max_items} more")


def _value_report(value: Any) -> OnnxValueReport:
    tensor_type = value.type.tensor_type
    dtype = _onnx_elem_type_name(tensor_type.elem_type)
    shape: list[Any] = []
    dynamic: list[dict[str, Any]] = []
    if tensor_type.HasField("shape"):
        for index, dim in enumerate(tensor_type.shape.dim):
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            elif dim.HasField("dim_param"):
                shape.append(dim.dim_param)
                dynamic.append({"index": index, "value": dim.dim_param})
            else:
                shape.append(None)
                dynamic.append({"index": index, "value": None})
    return OnnxValueReport(
        name=value.name,
        shape=shape,
        dtype=dtype,
        dynamic_dimensions=dynamic,
    )


def _initializer_report(onnx: Any, initializer: Any) -> OnnxInitializerReport:
    shape = list(initializer.dims)
    dtype = _onnx_elem_type_name(initializer.data_type)
    parameter_count = _numel(shape)
    itemsize = _tensor_type_itemsize(onnx, initializer.data_type)
    estimated_bytes = parameter_count * itemsize if itemsize is not None else len(initializer.raw_data)
    external = bool(initializer.external_data) or initializer.data_location == initializer.EXTERNAL
    return OnnxInitializerReport(
        name=initializer.name,
        shape=shape,
        dtype=dtype,
        parameter_count=parameter_count,
        estimated_bytes=estimated_bytes,
        estimated_size=format_file_size(estimated_bytes),
        external_data=external,
    )


def _operator_histogram(model: Any) -> list[dict[str, Any]]:
    counts: dict[tuple[str, str], int] = {}
    for node in model.graph.node:
        domain = node.domain or "ai.onnx"
        key = (domain, node.op_type)
        counts[key] = counts.get(key, 0) + 1
    return [
        {"domain": domain, "op_type": op_type, "count": count}
        for (domain, op_type), count in sorted(
            counts.items(), key=lambda item: (-item[1], item[0][0], item[0][1])
        )
    ]


def _custom_ops(model: Any) -> list[dict[str, str]]:
    seen: set[tuple[str, str]] = set()
    custom: list[dict[str, str]] = []
    for node in model.graph.node:
        domain = node.domain or "ai.onnx"
        if domain in STANDARD_ONNX_DOMAINS:
            continue
        key = (domain, node.op_type)
        if key in seen:
            continue
        seen.add(key)
        custom.append({"domain": domain, "op_type": node.op_type})
    return custom


def _dynamic_shape_warnings(values: list[OnnxValueReport]) -> list[str]:
    warnings = []
    for value in values:
        if value.dynamic_dimensions:
            dims = ", ".join(
                f"dim[{item['index']}]={item['value']}" for item in value.dynamic_dimensions
            )
            warnings.append(f"{value.name} has dynamic or unknown dimensions: {dims}")
    return warnings


def _tensorrt_risk_hints(
    *,
    dynamic_warnings: list[str],
    custom_ops: list[dict[str, str]],
    initializers: list[OnnxInitializerReport],
    dtype_histogram: dict[str, int],
) -> list[str]:
    hints: list[str] = []
    if dynamic_warnings:
        hints.append("TensorRT may require explicit min/opt/max shape profiles for dynamic inputs.")
    if custom_ops:
        hints.append("TensorRT may not support custom or non-standard ONNX domains without plugins.")
    large = [item for item in initializers if item.estimated_bytes >= LARGE_INITIALIZER_BYTES]
    if large:
        hints.append(
            "Large initializers may exceed edge memory budgets: "
            + ", ".join(f"{item.name}={item.estimated_size}" for item in large[:5])
        )
    if dtype_histogram and set(dtype_histogram) == {"FLOAT"}:
        hints.append("Initializers are FP32 only; consider FP16 export or target-side FP16 build if accuracy allows.")
    return hints


def _resolve_input_request(
    onnx_input: Any,
    *,
    spec_path: str | None,
    input_shape: list[int] | None,
    input_dtype: str | None,
    input_name: str | None,
) -> tuple[str, list[int], str]:
    spec_input = _load_spec_input(spec_path) if spec_path else None

    resolved_name = input_name
    if resolved_name is None and spec_input is not None:
        resolved_name = spec_input.get("name")
    if resolved_name is None:
        resolved_name = onnx_input.name
    if resolved_name != onnx_input.name:
        raise RuntimeError(
            f"input name '{resolved_name}' does not match ONNX model input '{onnx_input.name}'"
        )

    if input_shape is not None:
        resolved_shape = input_shape
    elif spec_input is not None:
        resolved_shape = _resolve_spec_shape(spec_input)
    else:
        resolved_shape = _resolve_onnx_shape(onnx_input.shape)

    if input_dtype is not None:
        resolved_dtype = input_dtype
    elif spec_input is not None:
        resolved_dtype = spec_input.get("dtype")
    else:
        resolved_dtype = ONNX_TYPE_TO_DTYPE.get(str(onnx_input.type))
        if resolved_dtype is None:
            print(
                f"Warning: could not infer dtype from ONNX type '{onnx_input.type}'; "
                "using float32"
            )
            resolved_dtype = "float32"

    if resolved_dtype not in DTYPE_TO_NUMPY:
        raise RuntimeError(f"unsupported input dtype for ONNX validation: {resolved_dtype}")
    return resolved_name, resolved_shape, resolved_dtype


def _load_spec_input(spec_path: str | None) -> dict[str, Any]:
    result = validate_spec_file(spec_path)
    if not result.valid or result.spec is None:
        errors = "; ".join(result.errors) or "spec did not load"
        raise RuntimeError(f"spec validation failed: {errors}")
    return result.spec["input"]


def _resolve_spec_shape(input_spec: dict[str, Any]) -> list[int]:
    shape = input_spec["shape"]
    if all(isinstance(dim, int) and not isinstance(dim, bool) for dim in shape):
        return list(shape)

    example_shape = input_spec.get("example_shape")
    if example_shape is None:
        dynamic_dims = [dim for dim in shape if isinstance(dim, str)]
        raise RuntimeError(
            "spec.input.shape contains dynamic dimensions "
            f"{dynamic_dims}; provide spec.input.example_shape or --input-shape"
        )
    if not isinstance(example_shape, list) or len(example_shape) != len(shape):
        raise RuntimeError("spec.input.example_shape must match spec.input.shape length")
    if not all(isinstance(dim, int) and not isinstance(dim, bool) for dim in example_shape):
        raise RuntimeError("spec.input.example_shape entries must be integers")
    return list(example_shape)


def _resolve_onnx_shape(shape: list[Any]) -> list[int]:
    resolved: list[int] = []
    for dim in shape:
        if isinstance(dim, int) and dim > 0:
            resolved.append(dim)
        else:
            raise RuntimeError(
                "ONNX input shape has dynamic or unknown dimensions; provide --input-shape"
            )
    return resolved


def _create_numpy_dummy_input(np: Any, shape: list[int], dtype_name: str) -> Any:
    dtype = np.dtype(DTYPE_TO_NUMPY[dtype_name])
    if np.issubdtype(dtype, np.floating):
        return np.random.randn(*shape).astype(dtype)
    return np.zeros(shape, dtype=dtype)


def _summarize_outputs(
    metadata: list[Any], values: list[Any], max_items: int
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for index, value in enumerate(values[:max_items]):
        name = metadata[index].name if index < len(metadata) else f"output_{index}"
        summary: dict[str, Any] = {
            "name": name,
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }
        if value.size and value.dtype.kind in {"f", "i", "u", "b"}:
            summary["min"] = float(value.min())
            summary["max"] = float(value.max())
            summary["mean"] = float(value.mean())
        summaries.append(summary)
    return summaries


def _looks_like_shape_resolution_error(text: str) -> bool:
    lowered = text.lower()
    return "dynamic" in lowered or "input shape" in lowered or "provide --input-shape" in lowered


def _onnx_elem_type_name(value: int) -> str:
    names = {
        1: "FLOAT",
        2: "UINT8",
        3: "INT8",
        4: "UINT16",
        5: "INT16",
        6: "INT32",
        7: "INT64",
        9: "BOOL",
        10: "FLOAT16",
        11: "DOUBLE",
        12: "UINT32",
        13: "UINT64",
        16: "BFLOAT16",
    }
    return names.get(value, f"UNKNOWN({value})")


def _tensor_type_itemsize(onnx: Any, data_type: int) -> int | None:
    try:
        np_dtype = onnx.helper.tensor_dtype_to_np_dtype(data_type)
        return int(np_dtype.itemsize)
    except Exception:
        sizes = {
            1: 4,
            2: 1,
            3: 1,
            4: 2,
            5: 2,
            6: 4,
            7: 8,
            9: 1,
            10: 2,
            11: 8,
            12: 4,
            13: 8,
            16: 2,
        }
        return sizes.get(data_type)


def _numel(shape: list[int]) -> int:
    total = 1
    for dim in shape:
        total *= int(dim)
    return total
