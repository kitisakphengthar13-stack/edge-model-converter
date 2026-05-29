from __future__ import annotations

from pathlib import Path

from .checkpoint import load_checkpoint_safe, load_checkpoint_unsafe
from .inspect_pt import build_inspection_report
from .onnx_validate import build_onnx_inspection_report
from .reports import (
    ReportEnvelope,
    envelope_from_onnx_report,
    envelope_from_pytorch_report,
    infer_artifact_kind,
)


def inspect_artifact(
    path: str | Path,
    *,
    unsafe_load: bool = False,
    input_shape: list[int] | None = None,
    max_items: int = 20,
    target: str = "generic",
) -> ReportEnvelope:
    artifact_path = Path(path)
    kind = infer_artifact_kind(artifact_path)
    if kind == "onnx":
        report = build_onnx_inspection_report(
            str(artifact_path),
            input_shape=input_shape,
            max_items=max_items,
            target=target,
        )
        return envelope_from_onnx_report(report.to_dict())
    if kind == "pytorch_checkpoint":
        load_result = (
            load_checkpoint_unsafe(artifact_path)
            if unsafe_load
            else load_checkpoint_safe(artifact_path)
        )
        checkpoint = load_result.checkpoint
        report = build_inspection_report(artifact_path, checkpoint, max_items=max_items)
        envelope = envelope_from_pytorch_report(report.to_dict())
        for warning in load_result.warnings:
            from .reports import ReportMessage

            envelope.warnings.append(ReportMessage("warning", warning, "load_warning"))
        return envelope
    raise RuntimeError(f"unsupported artifact type: {artifact_path}")
