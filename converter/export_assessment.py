from __future__ import annotations

from pathlib import Path
from typing import Any

from .checkpoint import load_checkpoint_safe, load_checkpoint_unsafe
from .inspect_pt import (
    build_inspection_report,
    collect_mapping_entries,
    detect_checkpoint_kind,
    detect_task_hints,
    find_state_dict,
    framework_from_unsupported_global,
)
from .spec import validate_spec_file

SUPPORTED_TARGET_FORMATS = {"onnx"}
EXECUTABLE_TOOLKIT_ONNX_STRATEGIES = {"full_model", "feature_extractor_only"}
EXECUTABLE_CHECKPOINT_LOAD_MODES = {"none", "safe_weights_only", "unsafe_trusted_local"}

FRAMEWORK_CAPABILITY_REGISTRY: dict[str, dict[str, Any]] = {
    "ultralytics": {
        "provider": "ultralytics",
        "official_exporter_family": True,
        "target_formats": {
            "onnx": {
                "status": "known_from_registry",
                "confidence": "high",
                "summary": "Ultralytics provides a source-library ONNX export route for supported models.",
            },
        },
        "preferred_route": "official_source_exporter",
    },
    "anomalib": {
        "provider": "anomalib",
        "official_exporter_family": True,
        "target_formats": {
            "onnx": {
                "status": "known_from_registry",
                "confidence": "high",
                "summary": "Anomalib provides a source-library ONNX export route for supported models.",
            },
        },
        "preferred_route": "official_source_exporter",
    },
    "custom_pytorch": {
        "provider": None,
        "official_exporter_family": False,
        "target_formats": {},
        "preferred_route": "toolkit_generic_exporter",
    },
    "unknown": {
        "provider": None,
        "official_exporter_family": False,
        "target_formats": {},
        "preferred_route": "manual_review",
    },
}


def assess_export_from_spec(
    spec_path: str, target_format: str | None = None
) -> dict[str, Any]:
    validation = validate_spec_file(spec_path)
    if not validation.valid or validation.spec is None:
        errors = "; ".join(validation.errors) or "spec did not load"
        raise RuntimeError(f"spec validation failed: {errors}")

    spec = validation.spec
    resolved_target = _resolve_target_format(target_format, spec)
    detected_source = _detect_source_from_spec(spec)
    official = _assess_official_exporter(detected_source, resolved_target, spec=spec)
    toolkit = _assess_toolkit_generic_exporter_from_spec(spec, resolved_target)
    recommendation = _recommend_route(detected_source, official, toolkit, resolved_target)

    return {
        "input_mode": "spec",
        "path": str(spec_path),
        "requested_target_format": resolved_target,
        "detected_source": detected_source,
        "official_source_exporter": official,
        "toolkit_generic_exporter": toolkit,
        "recommended_route": recommendation,
    }


def assess_export_from_model_path(
    model_path: str,
    target_format: str = "onnx",
    unsafe_load: bool = False,
    max_items: int = 80,
) -> dict[str, Any]:
    path = Path(model_path)
    if not path.is_file():
        raise RuntimeError(f"model/checkpoint file not found: {path}")

    resolved_target = _normalize_target_format(target_format)
    try:
        load_result = (
            load_checkpoint_unsafe(path) if unsafe_load else load_checkpoint_safe(path)
        )
    except Exception as exc:
        framework_hint = framework_from_unsupported_global(str(exc))
        if framework_hint is not None:
            detected_source = _source_from_framework_hint(path, framework_hint, str(exc))
            official = _assess_official_exporter(detected_source, resolved_target, spec=None)
            toolkit = _checkpoint_toolkit_unknown(resolved_target)
            return {
                "input_mode": "checkpoint",
                "path": str(path),
                "requested_target_format": resolved_target,
                "load_mode": "unsafe_trusted_local" if unsafe_load else "safe_weights_only",
                "load_warnings": [str(exc)],
                "detected_source": detected_source,
                "official_source_exporter": official,
                "toolkit_generic_exporter": toolkit,
                "recommended_route": _recommend_route(detected_source, official, toolkit, resolved_target),
            }
        mode = "unsafe" if unsafe_load else "safe"
        raise RuntimeError(
            f"{mode} checkpoint loading failed during preliminary assessment: {exc}"
        ) from exc

    checkpoint = load_result.checkpoint
    report = build_inspection_report(path, checkpoint, max_items=max_items)
    detected_source = _detect_source_from_checkpoint(path, checkpoint, max_items=max_items, report=report)
    official = _assess_official_exporter(detected_source, resolved_target, spec=None)
    toolkit = _checkpoint_toolkit_unknown(resolved_target)
    recommendation = _recommend_route(detected_source, official, toolkit, resolved_target)
    if recommendation["route"] == "toolkit_generic_exporter":
        recommendation = {
            "route": "manual_review",
            "confidence": "medium",
            "reason": [
                "checkpoint-only assessment cannot prove toolkit generic exporter feasibility",
                "create or refine a spec before attempting toolkit export",
            ],
        }

    return {
        "input_mode": "checkpoint",
        "path": str(path),
        "requested_target_format": resolved_target,
        "load_mode": "unsafe_trusted_local" if unsafe_load else "safe_weights_only",
        "load_warnings": load_result.warnings,
        "detected_source": detected_source,
        "official_source_exporter": official,
        "toolkit_generic_exporter": toolkit,
        "recommended_route": recommendation,
    }


def _resolve_target_format(target_format: str | None, spec: dict[str, Any]) -> str:
    if target_format:
        return _normalize_target_format(target_format)
    conversion = spec.get("conversion") if isinstance(spec.get("conversion"), dict) else {}
    declared = conversion.get("target_format")
    if isinstance(declared, str) and declared:
        return _normalize_target_format(declared)
    export = spec.get("export") if isinstance(spec.get("export"), dict) else {}
    declared = export.get("target_format")
    if isinstance(declared, str) and declared:
        return _normalize_target_format(declared)
    return "onnx"


def _normalize_target_format(value: str) -> str:
    normalized = value.lower().replace("_", "").replace("-", "")
    if normalized not in SUPPORTED_TARGET_FORMATS:
        supported = ", ".join(sorted(SUPPORTED_TARGET_FORMATS))
        raise RuntimeError(f"unsupported target format '{value}'. Supported: {supported}")
    return normalized


def _detect_source_from_spec(spec: dict[str, Any]) -> dict[str, Any]:
    source = spec.get("source") if isinstance(spec.get("source"), dict) else {}
    model = spec.get("model") if isinstance(spec.get("model"), dict) else {}
    framework = source.get("framework") or spec.get("framework")
    model_family = source.get("model_family") or model.get("architecture")
    task = spec.get("task")
    evidence: list[str] = []

    if isinstance(framework, str) and framework:
        normalized_framework = _normalize_framework(framework)
        evidence.append(f"spec declares framework '{framework}'")
        status = "declared"
        confidence = "high"
    elif _looks_like_ultralytics(model_family, task, model):
        normalized_framework = "ultralytics"
        evidence.append("spec model/task fields contain YOLO/Ultralytics-like hints")
        status = "inferred"
        confidence = "medium"
    elif _looks_like_anomalib(model_family, task, model):
        normalized_framework = "anomalib"
        evidence.append("spec model/task fields contain Anomalib/PatchCore-like hints")
        status = "inferred"
        confidence = "medium"
    elif model.get("module") and model.get("class_name"):
        normalized_framework = "custom_pytorch"
        evidence.append("spec provides model.module and model.class_name with no source framework declaration")
        status = "inferred"
        confidence = "medium"
    else:
        normalized_framework = "unknown"
        evidence.append("spec does not declare a known source framework")
        status = "unknown"
        confidence = "low"

    if not model_family:
        model_family = _infer_model_family_from_task(task)
    if isinstance(model_family, str) and model_family:
        evidence.append(f"model family/architecture hint is '{model_family}'")

    return {
        "framework": normalized_framework,
        "model_family": model_family or "unknown",
        "task": task or "unknown",
        "status": status,
        "confidence": confidence,
        "evidence": evidence,
    }


def _checkpoint_toolkit_unknown(target_format: str) -> dict[str, Any]:
    return {
        "status": "unknown",
        "target_format": target_format,
        "confidence": "low",
        "evidence": [
            "checkpoint-path assessment does not include model.module, model.class_name, input spec, or output spec"
        ],
        "blockers": [],
        "unknowns": [
            "model construction is unavailable without a spec",
            "checkpoint loading rule is not declared in a spec",
            "input/output contract is unavailable without a spec",
            "dry-run forward was not attempted",
        ],
    }


def _source_from_framework_hint(path: Path, hint: Any, error_text: str) -> dict[str, Any]:
    model_family = "unknown"
    task = "unknown"
    lower = error_text.lower()
    if hint.framework == "ultralytics":
        model_family = "yolo"
        if "segmentationmodel" in lower:
            task = "segmentation"
        elif "detectionmodel" in lower:
            task = "object_detection"
    elif hint.framework == "anomalib":
        task = "anomaly_detection"
        if "patchcore" in lower:
            model_family = "patchcore"
        elif "reverse_distillation" in lower or "reversedistillation" in lower:
            model_family = "reverse_distillation"
    return {
        "framework": hint.framework,
        "model_family": model_family,
        "task": task,
        "status": "inferred",
        "confidence": hint.confidence,
        "evidence": [
            f"safe checkpoint loading failed for {path.name} with framework-specific global",
            *hint.evidence,
        ],
    }


def _detect_source_from_checkpoint(
    path: Path, checkpoint: Any, max_items: int, report: Any | None = None
) -> dict[str, Any]:
    checkpoint_kind = detect_checkpoint_kind(checkpoint, path)
    state_dict_name, state_dict = find_state_dict(checkpoint)
    entries = collect_mapping_entries(checkpoint, max_keys=max_items * 10)
    key_names = [item.path for item in entries[: max_items * 10]]
    if state_dict is not None:
        key_names.extend(str(key) for key in list(state_dict.keys())[: max_items * 10])
    lowered_blob = " ".join(key.lower() for key in key_names[: max_items * 20])
    type_name = f"{type(checkpoint).__module__}.{type(checkpoint).__qualname__}".lower()
    task_hints = detect_task_hints(key_names)
    evidence = [
        f"checkpoint kind detected as {checkpoint_kind}",
        f"top-level object type is {type(checkpoint).__module__}.{type(checkpoint).__qualname__}",
    ]
    if state_dict_name:
        evidence.append(f"state_dict-like weights found at {state_dict_name}")

    framework = "unknown"
    model_family = "unknown"
    task = "unknown"
    status = "unknown"
    confidence = "low"

    if report and report.framework_hints:
        best_hint = sorted(
            report.framework_hints,
            key=lambda item: 0 if item.confidence == "high" else 1,
        )[0]
        framework = best_hint.framework
        status = "inferred"
        confidence = best_hint.confidence
        evidence.extend(best_hint.evidence)
        if framework == "ultralytics":
            model_family = "yolo"
            hint_evidence = " ".join(best_hint.evidence).lower()
            segmentation_high = any(
                task_name == "segmentation" and task_confidence == "high"
                for task_name, task_confidence, _ in task_hints
            )
            detection_present = any(
                task_name == "object_detection" for task_name, _, _ in task_hints
            )
            if "segmentationmodel" in hint_evidence or segmentation_high:
                task = "segmentation"
            elif "detectionmodel" in hint_evidence or detection_present:
                task = "object_detection"
        elif framework == "anomalib":
            task = "anomaly_detection"
            if "patchcore" in lowered_blob or "memory_bank" in lowered_blob:
                model_family = "patchcore"
            elif "reverse_distillation" in lowered_blob or "reversedistillation" in lowered_blob:
                model_family = "reverse_distillation"
    elif "ultralytics" in type_name or "ultralytics" in lowered_blob:
        framework = "ultralytics"
        model_family = "yolo" if "yolo" in lowered_blob or "yolo" in type_name else "unknown"
        status = "inferred"
        confidence = "high"
        evidence.append("checkpoint metadata/type contains 'ultralytics'")
    elif "yolo" in lowered_blob:
        framework = "ultralytics"
        model_family = "yolo"
        status = "inferred"
        confidence = "medium"
        evidence.append("checkpoint keys contain YOLO-like hints")
    elif _checkpoint_has_anomalib_hints(lowered_blob, task_hints):
        framework = "anomalib"
        model_family = "patchcore" if "patchcore" in lowered_blob or "memory_bank" in lowered_blob else "unknown"
        status = "inferred"
        confidence = "medium"
        evidence.append("checkpoint keys contain Anomalib/PatchCore-like hints")

    for task_name, task_confidence, matches in task_hints[:3]:
        evidence.append(
            f"task hint {task_name} has {task_confidence} confidence from keys: {', '.join(matches[:5])}"
        )
        if model_family == "unknown" and task_name == "anomaly_detection":
            model_family = "anomaly_detection"
        elif model_family == "unknown" and task_name == "object_detection":
            model_family = "object_detection"

    resolved_task = task if task != "unknown" else (task_hints[0][0] if task_hints else "unknown")
    return {
        "framework": framework,
        "model_family": model_family,
        "task": resolved_task,
        "status": status,
        "confidence": confidence,
        "evidence": evidence,
    }


def _assess_official_exporter(
    detected_source: dict[str, Any],
    target_format: str,
    *,
    spec: dict[str, Any] | None,
) -> dict[str, Any]:
    framework = detected_source["framework"]
    registry_entry = FRAMEWORK_CAPABILITY_REGISTRY.get(framework, FRAMEWORK_CAPABILITY_REGISTRY["unknown"])
    target_info = registry_entry.get("target_formats", {}).get(target_format)
    evidence: list[str] = []
    unknowns: list[str] = []

    declared = _declared_export_route(spec)
    if declared:
        evidence.append(f"spec declares export route/provider information: {declared}")

    if target_info:
        status = target_info["status"]
        confidence = target_info["confidence"]
        evidence.append(target_info["summary"])
        unknowns.extend(
            [
                "exporter execution was not attempted",
                "installed source-framework version/export compatibility is not verified",
                "model-specific operator and postprocessing export support is not verified",
            ]
        )
        if detected_source["status"] == "declared":
            evidence.append("source framework was declared by spec")
        elif detected_source["status"] == "inferred":
            evidence.append("source framework was inferred from hints; this is not verification")
            if confidence == "high":
                confidence = "medium"
                status = "likely"
    elif framework in {"custom_pytorch", "unknown"}:
        status = "unknown" if framework == "unknown" else "not_applicable"
        confidence = "low"
        evidence.append("no known official source-library exporter is registered for this source")
        unknowns.append("a source-specific exporter could still exist outside the current registry")
    else:
        status = "unknown"
        confidence = "low"
        evidence.append(
            f"registry has no official {target_format} export route for framework '{framework}'"
        )
        unknowns.append("a route may exist in a framework version or plugin not represented by this registry")

    return {
        "status": status,
        "provider": registry_entry.get("provider"),
        "target_format": target_format,
        "confidence": confidence,
        "evidence": evidence,
        "blockers": [],
        "unknowns": unknowns,
    }


def _assess_toolkit_generic_exporter_from_spec(
    spec: dict[str, Any], target_format: str
) -> dict[str, Any]:
    if target_format != "onnx":
        return {
            "status": "not_applicable",
            "target_format": target_format,
            "confidence": "high",
            "evidence": ["toolkit generic exporter currently targets ONNX only"],
            "blockers": [f"generic toolkit export to {target_format} is not implemented"],
            "unknowns": [],
        }

    checkpoint = spec.get("checkpoint") if isinstance(spec.get("checkpoint"), dict) else {}
    model = spec.get("model") if isinstance(spec.get("model"), dict) else {}
    conversion = spec.get("conversion") if isinstance(spec.get("conversion"), dict) else {}
    input_spec = spec.get("input") if isinstance(spec.get("input"), dict) else {}
    output_spec = spec.get("output") if isinstance(spec.get("output"), dict) else {}
    blockers: list[str] = []
    evidence: list[str] = []
    unknowns = ["dry-run forward was not attempted by assess-export"]

    if model.get("module") and model.get("class_name"):
        evidence.append("model.module and model.class_name are present")
    else:
        blockers.append("model.module and/or model.class_name are missing")

    load_mode = checkpoint.get("load_mode")
    if load_mode in EXECUTABLE_CHECKPOINT_LOAD_MODES:
        evidence.append(f"checkpoint.load_mode '{load_mode}' is executable by current toolkit rules")
    else:
        blockers.append(f"checkpoint.load_mode '{load_mode}' is not executable by export-onnx")

    strategy = conversion.get("strategy")
    if strategy in EXECUTABLE_TOOLKIT_ONNX_STRATEGIES:
        evidence.append(f"conversion.strategy '{strategy}' is supported by export-onnx")
    else:
        blockers.append(f"conversion.strategy '{strategy}' is not executable by export-onnx")

    if _input_ready(input_spec):
        evidence.append("input spec has name, dtype, and shape")
    else:
        blockers.append("input spec is incomplete")

    if _output_ready(output_spec):
        evidence.append("output spec has at least one output name")
    else:
        blockers.append("output spec is incomplete")

    if conversion.get("wrapper"):
        blockers.append("conversion.wrapper is declared but wrapper execution is not implemented")
    if checkpoint.get("loader"):
        blockers.append("checkpoint.loader is declared but custom loader execution is not implemented")

    if blockers:
        status = "blocked"
        confidence = "high"
    else:
        status = "possible"
        confidence = "medium"
        unknowns.extend(
            [
                "model import/instantiation was not attempted",
                "checkpoint file existence/loading was not attempted",
                "ONNX export success is not guaranteed until export-onnx runs",
            ]
        )

    return {
        "status": status,
        "target_format": "onnx",
        "confidence": confidence,
        "evidence": evidence,
        "blockers": blockers,
        "unknowns": unknowns,
    }


def _recommend_route(
    detected_source: dict[str, Any],
    official: dict[str, Any],
    toolkit: dict[str, Any],
    target_format: str,
) -> dict[str, Any]:
    official_status = official["status"]
    official_known = official_status in {"known_from_registry", "declared_by_spec", "likely"}
    if official_known and detected_source["framework"] in {"ultralytics", "anomalib"}:
        reason = [
            f"{detected_source['framework']} is a known source-first framework for {target_format} planning",
            "official source-library exporters should be preferred before toolkit fallback",
        ]
        if toolkit["status"] == "blocked":
            reason.append("toolkit generic exporter is currently blocked for this spec")
        return {
            "route": "official_source_exporter",
            "confidence": "high" if detected_source["status"] == "declared" else "medium",
            "reason": reason,
        }

    if target_format == "onnx" and toolkit["status"] == "possible":
        return {
            "route": "toolkit_generic_exporter",
            "confidence": "medium",
            "reason": [
                "no preferred official source exporter was identified",
                "spec contains the fields required to attempt generic PyTorch-to-ONNX export",
            ],
        }

    reason = ["no executable or strongly recommended export route was proven"]
    if toolkit.get("blockers"):
        reason.extend(toolkit["blockers"])
    if official_status in {"unknown", "not_applicable"}:
        reason.append("official source exporter availability is unknown or not applicable")
    return {
        "route": "manual_review",
        "confidence": "medium",
        "reason": reason,
    }


def _normalize_framework(value: str) -> str:
    normalized = value.lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "yolo": "ultralytics",
        "ultralytics_yolo": "ultralytics",
        "pytorch": "custom_pytorch",
        "torch": "custom_pytorch",
        "custom": "custom_pytorch",
    }
    return aliases.get(normalized, normalized)


def _looks_like_ultralytics(model_family: Any, task: Any, model: dict[str, Any]) -> bool:
    blob = " ".join(str(value).lower() for value in (model_family, task, model.get("architecture")))
    return "ultralytics" in blob or "yolo" in blob


def _looks_like_anomalib(model_family: Any, task: Any, model: dict[str, Any]) -> bool:
    blob = " ".join(str(value).lower() for value in (model_family, task, model.get("architecture")))
    return any(keyword in blob for keyword in ("anomalib", "patchcore", "anomaly_detection"))


def _checkpoint_has_anomalib_hints(
    lowered_blob: str, task_hints: list[tuple[str, str, list[str]]]
) -> bool:
    if any(keyword in lowered_blob for keyword in ("anomalib", "patchcore", "memory_bank")):
        return True
    return any(task_name == "anomaly_detection" for task_name, _, _ in task_hints)


def _infer_model_family_from_task(task: Any) -> str | None:
    if not isinstance(task, str):
        return None
    if "yolo" in task.lower():
        return "yolo"
    return None


def _declared_export_route(spec: dict[str, Any] | None) -> str | None:
    if not spec:
        return None
    export = spec.get("export") if isinstance(spec.get("export"), dict) else {}
    parts = []
    for key in ("route", "provider", "capability_status"):
        value = export.get(key)
        if isinstance(value, str) and value:
            parts.append(f"{key}={value}")
    return ", ".join(parts) if parts else None


def _input_ready(input_spec: dict[str, Any]) -> bool:
    shape = input_spec.get("shape")
    return (
        isinstance(input_spec.get("name"), str)
        and isinstance(input_spec.get("dtype"), str)
        and isinstance(shape, list)
        and bool(shape)
    )


def _output_ready(output_spec: dict[str, Any]) -> bool:
    names = output_spec.get("names")
    return isinstance(names, list) and bool(names) and all(isinstance(name, str) for name in names)
