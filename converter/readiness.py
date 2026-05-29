from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


SUPPORTED_READINESS_PROFILES = (
    "generic",
    "tensorrt-orin-nano",
    "onnxruntime-cpu",
    "openvino",
)

WARNING_INITIALIZER_BYTES = 512 * 1024 * 1024
ERROR_INITIALIZER_BYTES_TENSORRT_ORIN_NANO = 2 * 1024 * 1024 * 1024
WARNING_PARAMETER_COUNT = 100_000_000
HIGH_WARNING_PARAMETER_COUNT = 500_000_000


@dataclass
class ReadinessFinding:
    severity: str
    code: str
    message: str
    recommendation: str | None = None


@dataclass
class ReadinessReport:
    target: str
    score: int
    readiness_level: str
    findings: list[ReadinessFinding] = field(default_factory=list)
    recommended_next_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_readiness_report(
    report: dict[str, Any],
    *,
    target: str = "generic",
    requested_precision: str | None = None,
) -> ReadinessReport:
    if target not in SUPPORTED_READINESS_PROFILES:
        supported = ", ".join(SUPPORTED_READINESS_PROFILES)
        raise RuntimeError(f"unsupported readiness target '{target}'. Supported targets: {supported}")

    findings: list[ReadinessFinding] = []
    edge_target = target != "generic"
    tensorrt_target = target.startswith("tensorrt")

    validation = report.get("validation", {})
    if not validation.get("checker_passed"):
        _add(
            findings,
            "error",
            "onnx_checker_failed",
            validation.get("checker_error") or "ONNX checker failed.",
            "Fix or re-export the ONNX artifact before deployment planning.",
        )
    if not validation.get("ort_session_created"):
        _add(
            findings,
            "error",
            "onnxruntime_session_failed",
            validation.get("ort_error") or "ONNX Runtime could not create a session.",
            "Validate runtime dependencies and model compatibility before deployment.",
        )
    if validation.get("inference_error"):
        _add(
            findings,
            "error",
            "onnxruntime_inference_failed",
            validation["inference_error"],
            "Run dummy inference with a known-good input shape and fix the failing graph/runtime issue.",
        )
    if validation.get("inference_skipped_reason"):
        _add(
            findings,
            "warning",
            "inference_not_run",
            f"Dummy inference was skipped: {validation['inference_skipped_reason']}",
            "Provide --input-shape for dynamic or unknown ONNX inputs.",
        )

    dynamic_warnings = report.get("dynamic_shape_warnings", [])
    if tensorrt_target and dynamic_warnings:
        _add(
            findings,
            "warning",
            "tensorrt_dynamic_shapes",
            "TensorRT targets require explicit min/opt/max profiles for dynamic dimensions.",
            "Plan TensorRT builds with concrete min/opt/max shape profiles.",
        )
    elif dynamic_warnings:
        _add(
            findings,
            "info",
            "dynamic_shapes",
            "The model contains dynamic or symbolic dimensions.",
            "Confirm target runtime shape-profile handling before deployment.",
        )

    if report.get("custom_ops"):
        _add(
            findings,
            "error" if tensorrt_target else "warning",
            "custom_ops",
            "The model contains custom or non-standard ONNX operator domains.",
            "Verify runtime plugin support or re-export with standard operators only.",
        )

    initializer_bytes = int(report.get("estimated_initializer_bytes") or 0)
    if tensorrt_target and initializer_bytes > ERROR_INITIALIZER_BYTES_TENSORRT_ORIN_NANO:
        _add(
            findings,
            "error",
            "initializer_memory_over_2gb",
            "Initializer memory exceeds 2 GB on tensorrt-orin-nano.",
            "Reduce model size, use a different architecture, or deploy on a target with a larger memory budget.",
        )
    elif initializer_bytes > WARNING_INITIALIZER_BYTES:
        _add(
            findings,
            "warning",
            "initializer_memory_over_512mb",
            "Initializer memory exceeds 512 MB.",
            "Check target memory budget and consider compression, pruning, or a smaller model.",
        )

    parameter_count = int(report.get("parameter_count") or 0)
    if parameter_count > HIGH_WARNING_PARAMETER_COUNT:
        _add(
            findings,
            "warning",
            "params_over_500m",
            "Parameter count exceeds 500M.",
            "Treat this as a high-risk edge deployment unless the target has a large memory budget.",
        )
    elif parameter_count > WARNING_PARAMETER_COUNT:
        _add(
            findings,
            "warning",
            "params_over_100m",
            "Parameter count exceeds 100M.",
            "Check latency, memory, and model-loading constraints on the target device.",
        )

    if edge_target and report.get("has_external_data"):
        _add(
            findings,
            "warning",
            "external_data",
            "The ONNX model uses external data files.",
            "Package and validate the complete ONNX directory, not only the .onnx file.",
        )

    dtype_histogram = report.get("initializer_dtype_histogram") or {}
    if edge_target and dtype_histogram and set(dtype_histogram) == {"FLOAT"}:
        _add(
            findings,
            "warning",
            "fp32_only",
            "Initializers are FP32 only on an edge deployment target.",
            "Consider FP16 export or target-side FP16 build if accuracy allows.",
        )

    if requested_precision == "int8" and not _has_int8_evidence(report):
        _add(
            findings,
            "warning",
            "int8_without_quantization_evidence",
            "INT8 was requested but no INT8 initializer or quantization evidence was found.",
            "Provide calibration or an explicitly quantized model before treating INT8 as deployment-ready.",
        )

    if _looks_like_segmentation_outputs(report.get("outputs", [])):
        _add(
            findings,
            "warning",
            "segmentation_outputs",
            "The model has segmentation-like multi-output tensors that may increase output memory and postprocessing cost.",
            "Budget output tensor memory and postprocessing latency on the target runtime.",
        )

    if _looks_like_patchcore_memory_bank(report):
        _add(
            findings,
            "error" if tensorrt_target else "warning",
            "patchcore_memory_bank",
            "Very large PatchCore/anomaly memory-bank artifacts were detected.",
            "Consider memory-bank reduction, alternative anomaly architecture, or non-edge deployment.",
        )

    errors = [item for item in findings if item.severity == "error"]
    warnings = [item for item in findings if item.severity == "warning"]
    level = "blocked" if errors else "caution" if warnings else "pass"
    score = max(0, 100 - len(warnings) * 10 - len(errors) * 30)
    actions = _recommended_actions(findings, target)
    return ReadinessReport(
        target=target,
        score=score,
        readiness_level=level,
        findings=findings,
        recommended_next_actions=actions,
    )


def _add(
    findings: list[ReadinessFinding],
    severity: str,
    code: str,
    message: str,
    recommendation: str | None = None,
) -> None:
    findings.append(
        ReadinessFinding(
            severity=severity,
            code=code,
            message=message,
            recommendation=recommendation,
        )
    )


def _has_int8_evidence(report: dict[str, Any]) -> bool:
    dtype_histogram = report.get("initializer_dtype_histogram") or {}
    if any(dtype in dtype_histogram for dtype in {"INT8", "UINT8"}):
        return True
    operators = report.get("operator_histogram") or []
    quant_ops = {"QuantizeLinear", "DequantizeLinear", "QLinearConv", "QLinearMatMul"}
    return any(item.get("op_type") in quant_ops for item in operators)


def _looks_like_segmentation_outputs(outputs: list[dict[str, Any]]) -> bool:
    if len(outputs) < 2:
        return False
    for output in outputs:
        shape = output.get("shape") or []
        if len(shape) == 4:
            return True
    return False


def _looks_like_patchcore_memory_bank(report: dict[str, Any]) -> bool:
    for item in report.get("largest_initializers", []) or []:
        name = str(item.get("name", "")).lower()
        estimated_bytes = int(item.get("estimated_bytes") or 0)
        if "memory_bank" in name and estimated_bytes > WARNING_INITIALIZER_BYTES:
            return True
    return False


def _recommended_actions(findings: list[ReadinessFinding], target: str) -> list[str]:
    actions: list[str] = []
    for finding in findings:
        if finding.recommendation and finding.recommendation not in actions:
            actions.append(finding.recommendation)
    if not actions:
        actions.append(f"No blocking edge-readiness issues were detected for {target}.")
    return actions
