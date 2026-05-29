from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


SEVERITY_RANK = {"info": 0, "warning": 1, "error": 2}


@dataclass
class ReportMessage:
    severity: str
    message: str
    code: str | None = None


@dataclass
class ReportEnvelope:
    artifact_path: str
    artifact_kind: str
    file_metadata: dict[str, Any]
    inspection_status: str
    report: dict[str, Any] | None = None
    warnings: list[ReportMessage] = field(default_factory=list)
    errors: list[ReportMessage] = field(default_factory=list)
    framework_hints: list[dict[str, Any]] = field(default_factory=list)
    readiness_hints: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "artifact_path": self.artifact_path,
            "artifact_kind": self.artifact_kind,
            "file_metadata": self.file_metadata,
            "inspection_status": self.inspection_status,
            "warnings": [message.__dict__ for message in self.warnings],
            "errors": [message.__dict__ for message in self.errors],
            "framework_hints": self.framework_hints,
            "readiness_hints": self.readiness_hints,
            "report": self.report,
        }


def infer_artifact_kind(path: str | Path) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".pt", ".pth", ".ckpt"}:
        return "pytorch_checkpoint"
    return "unsupported"


def messages_reach_threshold(
    warnings: list[ReportMessage],
    errors: list[ReportMessage],
    threshold: str | None,
) -> bool:
    if threshold is None:
        return False
    threshold_rank = SEVERITY_RANK[threshold]
    return any(
        SEVERITY_RANK[item.severity] >= threshold_rank
        for item in [*warnings, *errors]
    )


def envelope_from_pytorch_report(report: dict[str, Any]) -> ReportEnvelope:
    warnings = [
        ReportMessage("warning", item, "deployment_signal")
        for item in report.get("deployment_signals", [])
    ]
    framework_hints = report.get("framework_hints", [])
    return ReportEnvelope(
        artifact_path=report["path"],
        artifact_kind="pytorch_checkpoint",
        file_metadata={
            "size_bytes": report["size_bytes"],
            "size": report["size"],
            "top_level_type": report["top_level_type"],
            "checkpoint_kind": report["checkpoint_kind"],
        },
        inspection_status="ok",
        warnings=warnings,
        framework_hints=framework_hints,
        readiness_hints=report.get("deployment_signals", []),
        report=report,
    )


def envelope_from_onnx_report(report: dict[str, Any]) -> ReportEnvelope:
    readiness = report.get("readiness") or {}
    readiness_codes = {
        str(finding.get("code", ""))
        for finding in readiness.get("findings", [])
    }
    warnings = [
        ReportMessage("warning", item, "dynamic_shape")
        for item in report.get("dynamic_shape_warnings", [])
    ]
    warnings.extend(
        ReportMessage("warning", item, "tensorrt_risk")
        for item in report.get("tensorrt_risk_hints", [])
        if not _is_superseded_tensorrt_hint(item, readiness_codes)
    )
    errors = []
    validation_errors = []
    validation = report.get("validation", {})
    if not validation.get("checker_passed"):
        validation_errors.append(
            ReportMessage(
                "error",
                validation.get("checker_error") or "ONNX checker failed",
                "onnx_checker",
            )
        )
    if not validation.get("ort_session_created"):
        validation_errors.append(
            ReportMessage(
                "error",
                validation.get("ort_error") or "ONNX Runtime session failed",
                "onnxruntime",
            )
        )
    if validation.get("inference_error"):
        validation_errors.append(
            ReportMessage("error", validation["inference_error"], "onnxruntime_inference")
        )
    errors.extend(validation_errors)
    for finding in readiness.get("findings", []):
        message = ReportMessage(
            finding.get("severity", "warning"),
            finding.get("message", ""),
            finding.get("code"),
        )
        if message.severity == "error":
            errors.append(message)
        else:
            warnings.append(message)
    status = "error" if validation_errors else "ok"
    return ReportEnvelope(
        artifact_path=report["path"],
        artifact_kind="onnx",
        file_metadata={
            "size_bytes": report["size_bytes"],
            "size": report["size"],
            "ir_version": report["ir_version"],
            "producer_name": report["producer_name"],
            "graph_name": report["graph_name"],
        },
        inspection_status=status,
        warnings=warnings,
        errors=errors,
        readiness_hints=readiness.get("recommended_next_actions", report.get("tensorrt_risk_hints", [])),
        report=report,
    )


def _is_superseded_tensorrt_hint(message: str, readiness_codes: set[str]) -> bool:
    lowered = message.lower()
    if "fp32" in lowered and "fp32_only" in readiness_codes:
        return True
    if "min/opt/max" in lowered and "tensorrt_dynamic_shapes" in readiness_codes:
        return True
    if "large initializer" in lowered and (
        "initializer_memory_over_2gb" in readiness_codes
        or "initializer_memory_over_512mb" in readiness_codes
        or "patchcore_memory_bank" in readiness_codes
    ):
        return True
    if "custom" in lowered and "custom_ops" in readiness_codes:
        return True
    return False


def error_envelope(path: str | Path, message: str) -> ReportEnvelope:
    artifact_path = str(path)
    file_metadata: dict[str, Any] = {}
    if Path(path).is_file():
        size = Path(path).stat().st_size
        file_metadata = {"size_bytes": size}
    return ReportEnvelope(
        artifact_path=artifact_path,
        artifact_kind=infer_artifact_kind(path),
        file_metadata=file_metadata,
        inspection_status="error",
        errors=[ReportMessage("error", message, "inspection_failed")],
    )


def render_markdown_envelope(envelope: ReportEnvelope) -> str:
    report = envelope.report or {}
    lines = [
        f"# Inspection Report",
        "",
        f"- Artifact: `{envelope.artifact_path}`",
        f"- Kind: `{envelope.artifact_kind}`",
        f"- Status: `{envelope.inspection_status}`",
        f"- Size: `{envelope.file_metadata.get('size', envelope.file_metadata.get('size_bytes', '-'))}`",
        "",
    ]
    if envelope.errors or envelope.warnings:
        lines.extend(["## Messages", ""])
        for item in envelope.errors + envelope.warnings:
            code = f" `{item.code}`" if item.code else ""
            lines.append(f"- **{item.severity}**{code}: {item.message}")
        lines.append("")

    readiness = report.get("readiness") or {}
    if readiness:
        lines.extend(
            [
                "## Edge Readiness",
                "",
                f"- Target: `{readiness.get('target', '-')}`",
                f"- Level: `{readiness.get('readiness_level', '-')}`",
                f"- Score: `{readiness.get('score', '-')}`",
                "",
            ]
        )
        findings = readiness.get("findings", [])
        if findings:
            lines.append("### Findings")
            lines.append("")
            for item in findings:
                lines.append(
                    f"- **{item.get('severity', '-')}** `{item.get('code', '-')}`: "
                    f"{item.get('message', '-')}"
                )
            lines.append("")
        actions = readiness.get("recommended_next_actions", [])
        if actions:
            lines.append("### Recommended Next Actions")
            lines.append("")
            lines.extend(f"- {item}" for item in actions)
            lines.append("")

    if envelope.artifact_kind == "pytorch_checkpoint":
        lines.extend(_render_pytorch_markdown(report))
    elif envelope.artifact_kind == "onnx":
        lines.extend(_render_onnx_markdown(report))
    return "\n".join(lines).rstrip() + "\n"


def render_scan_markdown(envelopes: list[ReportEnvelope]) -> str:
    lines = [
        "# Batch Scan Report",
        "",
        "| File | Kind | Status | Readiness | Framework | Task | Params | Ops | Warnings | Errors |",
        "|---|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for envelope in envelopes:
        summary = summarize_envelope(envelope)
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{summary['file']}`",
                    str(summary["kind"]),
                    str(summary["status"]),
                    str(summary["readiness"] or "-"),
                    str(summary["framework"] or "-"),
                    str(summary["task"] or "-"),
                    str(summary["params"] if summary["params"] is not None else "-"),
                    str(summary["ops"] if summary["ops"] is not None else "-"),
                    str(summary["warnings"]),
                    str(summary["errors"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def summarize_envelope(envelope: ReportEnvelope) -> dict[str, Any]:
    report = envelope.report or {}
    framework = None
    task = None
    params = None
    ops = None
    readiness = None
    if envelope.artifact_kind == "pytorch_checkpoint":
        hints = report.get("framework_hints", [])
        framework = hints[0]["framework"] if hints else None
        task_hints = report.get("task_hints", [])
        task = task_hints[0]["task"] if task_hints else None
        params = report.get("tensor_summary", {}).get("parameter_count")
    elif envelope.artifact_kind == "onnx":
        params = report.get("parameter_count")
        ops = report.get("graph_summary", {}).get("node_count")
        readiness = (report.get("readiness") or {}).get("readiness_level")
    return {
        "file": envelope.artifact_path,
        "kind": envelope.artifact_kind,
        "status": envelope.inspection_status,
        "readiness": readiness,
        "framework": framework,
        "task": task,
        "params": params,
        "ops": ops,
        "warnings": len(envelope.warnings),
        "errors": len(envelope.errors),
    }


def _render_pytorch_markdown(report: dict[str, Any]) -> list[str]:
    tensor = report.get("tensor_summary", {})
    lines = [
        "## PyTorch Checkpoint",
        "",
        f"- Checkpoint kind: `{report.get('checkpoint_kind', '-')}`",
        f"- Top-level type: `{report.get('top_level_type', '-')}`",
        f"- Top-level key count: `{report.get('top_level_key_count', '-')}`",
        f"- Tensor count: `{tensor.get('tensor_count', 0)}`",
        f"- Parameter count: `{tensor.get('parameter_count', 0)}`",
        "",
    ]
    keys = report.get("top_level_keys", [])
    if keys:
        lines.extend(["## Top-Level Keys", ""])
        lines.extend(f"- `{key}`" for key in keys[:40])
        lines.append("")
    return lines


def _render_onnx_markdown(report: dict[str, Any]) -> list[str]:
    graph = report.get("graph_summary", {})
    lines = [
        "## ONNX Graph",
        "",
        f"- IR version: `{report.get('ir_version', '-')}`",
        f"- Producer: `{report.get('producer_name', '-')}` `{report.get('producer_version', '-')}`",
        f"- Node count: `{graph.get('node_count', 0)}`",
        f"- Initializer count: `{graph.get('initializer_count', 0)}`",
        f"- Parameter count: `{report.get('parameter_count', 0)}`",
        f"- Estimated initializer memory: `{report.get('estimated_initializer_size', '-')}`",
        "",
        "## Operators",
        "",
    ]
    operators = report.get("operator_histogram", [])
    if operators:
        lines.extend(
            f"- `{item['domain']}::{item['op_type']}`: {item['count']}"
            for item in operators
        )
    else:
        lines.append("- None")
    lines.append("")
    return lines
