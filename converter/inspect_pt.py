from __future__ import annotations

from dataclasses import asdict, dataclass, field
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .utils import (
    format_file_size,
    is_tensor_like,
    print_section,
    safe_dtype,
    safe_shape,
    safe_tensor_nbytes,
    truncate_repr,
)

STATE_DICT_KEYS = (
    "state_dict",
    "model_state_dict",
    "module_state_dict",
    "ema_state_dict",
)

METADATA_KEYS = (
    "hyper_parameters",
    "epoch",
    "global_step",
    "pytorch-lightning_version",
    "callbacks",
    "optimizer_states",
    "lr_schedulers",
    "config",
    "args",
    "model_args",
    "class_names",
    "names",
    "nc",
    "num_classes",
)

TASK_KEYWORDS = {
    "classification": (
        "classifier",
        "classification",
        "class_names",
        "num_classes",
        "fc.weight",
        "logits",
        "softmax",
    ),
    "object_detection": (
        "anchor",
        "anchors",
        "bbox",
        "box",
        "boxes",
        "detect",
        "detection",
        "rpn",
        "roi_heads",
        "yolo",
        "nms",
    ),
    "segmentation": (
        "seg",
        "segmentation",
        "mask",
        "masks",
        "decode_head",
        "mask_head",
        "seg_head",
    ),
    "pose_estimation": (
        "pose",
        "keypoint",
        "keypoints",
        "kpt",
        "heatmap",
        "joints",
    ),
    "anomaly_detection": (
        "anomaly",
        "anomalib",
        "stfpm",
        "padim",
        "patchcore",
        "fastflow",
        "cflow",
        "memory_bank",
        "anomaly_map",
        "image_threshold",
        "pixel_threshold",
        "post_processor",
        "image_min",
        "image_max",
        "pixel_min",
        "pixel_max",
        "feature_extractor",
    ),
    "autoencoder": (
        "autoencoder",
        "encoder",
        "decoder",
        "bottleneck",
        "latent",
        "vae",
    ),
    "ocr": (
        "ocr",
        "ctc",
        "charset",
        "text_recogn",
        "crnn",
        "recognition_head",
    ),
    "depth_estimation": (
        "depth",
        "disparity",
        "inverse_depth",
        "depth_head",
    ),
    "super_resolution": (
        "super_resolution",
        "superres",
        "sr.",
        "upsample",
        "upscale",
        "pixel_shuffle",
        "esrgan",
    ),
    "embedding_or_metric_learning": (
        "embedding",
        "embedder",
        "metric",
        "projection",
        "proj_head",
        "triplet",
        "contrastive",
    ),
}

ANOMALY_STRONG_KEYS = {
    "memory_bank",
    "anomaly_map",
    "image_threshold",
    "pixel_threshold",
    "post_processor",
    "image_min",
    "image_max",
    "pixel_min",
    "pixel_max",
}

THRESHOLD_KEYWORDS = (
    "threshold",
    "image_threshold",
    "pixel_threshold",
    "image_min",
    "image_max",
    "pixel_min",
    "pixel_max",
)

BACKBONE_LAYER_KEYWORDS = (
    "backbone",
    "feature_extractor",
    "layers",
    "layer",
    "blocks",
    "encoder",
)

LARGE_TENSOR_BYTES = 1024 * 1024
NESTED_INSPECTION_KEYS = (
    "model",
    "ema",
    "module",
    "state_dict",
    "model_state_dict",
    "hyper_parameters",
    "train_args",
    "train_metrics",
)


@dataclass
class TensorSummary:
    tensor_count: int = 0
    parameter_count: int = 0
    estimated_bytes: int = 0
    dtype_histogram: dict[str, int] = field(default_factory=dict)
    largest_tensors: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class NestedSummary:
    key: str
    type_name: str
    mapping_keys: list[str] = field(default_factory=list)
    mapping_key_count: int | None = None
    state_dict_like: bool = False
    tensor_summary: TensorSummary | None = None


@dataclass
class FrameworkHint:
    framework: str
    confidence: str
    evidence: list[str] = field(default_factory=list)


@dataclass
class InspectionReport:
    path: str
    size_bytes: int
    size: str
    top_level_type: str
    checkpoint_kind: str
    top_level_key_count: int | None = None
    top_level_keys: list[str] = field(default_factory=list)
    metadata_keys: list[str] = field(default_factory=list)
    state_dict_source: str | None = None
    state_dict_entries: int | None = None
    state_dict_preview: list[dict[str, Any]] = field(default_factory=list)
    tensor_summary: TensorSummary = field(default_factory=TensorSummary)
    nested: list[NestedSummary] = field(default_factory=list)
    framework_hints: list[FrameworkHint] = field(default_factory=list)
    deployment_signals: list[str] = field(default_factory=list)
    task_hints: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def inspect_checkpoint(path: str | Path, checkpoint: Any, max_items: int = 40) -> None:
    report = build_inspection_report(path, checkpoint, max_items=max_items)
    print_inspection_report(report, max_items=max_items)


def build_inspection_report(
    path: str | Path, checkpoint: Any, max_items: int = 40
) -> InspectionReport:
    checkpoint_path = Path(path)
    state_dict_name, state_dict = find_state_dict(checkpoint)
    key_names = [item.path for item in collect_mapping_entries(checkpoint)]
    if state_dict is not None:
        key_names.extend(str(key) for key in state_dict.keys())

    top_level_keys: list[str] = []
    top_level_key_count = None
    if isinstance(checkpoint, Mapping):
        top_level_keys = [str(key) for key in list(checkpoint.keys())[:max_items]]
        top_level_key_count = len(checkpoint)

    report = InspectionReport(
        path=str(checkpoint_path),
        size_bytes=checkpoint_path.stat().st_size,
        size=format_file_size(checkpoint_path.stat().st_size),
        top_level_type=_type_name(checkpoint),
        checkpoint_kind=detect_checkpoint_kind(checkpoint, checkpoint_path),
        top_level_key_count=top_level_key_count,
        top_level_keys=top_level_keys,
        metadata_keys=[key for key in METADATA_KEYS if isinstance(checkpoint, Mapping) and key in checkpoint],
        state_dict_source=state_dict_name or None,
        state_dict_entries=len(state_dict) if state_dict is not None else None,
        state_dict_preview=_state_dict_preview(state_dict, max_items) if state_dict is not None else [],
        tensor_summary=summarize_tensors(checkpoint, state_dict=state_dict, max_items=max_items),
        nested=summarize_nested(checkpoint, max_items=max_items),
        framework_hints=detect_framework_hints(checkpoint, key_names),
        deployment_signals=deployment_signals(checkpoint, state_dict),
        task_hints=[
            {"task": task, "confidence": confidence, "matches": matches[:max_items]}
            for task, confidence, matches in detect_task_hints(key_names)
        ],
    )
    return report


def print_inspection_report(report: InspectionReport, max_items: int = 40) -> None:
    print_section("File")
    print(f"Path: {report.path}")
    print(f"Size: {report.size}")
    print(f"Top-level Python object type: {report.top_level_type}")
    print(f"Detected checkpoint kind: {report.checkpoint_kind}")

    if report.top_level_key_count is not None:
        print_section("Top-level keys")
        print(f"Key count: {report.top_level_key_count}")
        print(f"Showing first {min(max_items, len(report.top_level_keys))} keys")
        for key in report.top_level_keys[:max_items]:
            print(f"- {key}")

    if report.metadata_keys:
        print_section("Common metadata")
        for key in report.metadata_keys:
            print(f"- {key}")

    if report.state_dict_source is not None:
        print_section("State dict")
        print(f"Source: {report.state_dict_source}")
        print(f"Entries: {report.state_dict_entries}")
        print(f"Showing first {min(max_items, len(report.state_dict_preview))} entries")
        for item in report.state_dict_preview[:max_items]:
            print(
                f"- {item['key']}: tensor_like={item['tensor_like']}, "
                f"shape={item['shape']}, dtype={item['dtype']}"
            )

    print_tensor_summary(report.tensor_summary)
    print_nested_summary(report.nested, max_items=max_items)
    print_framework_hints(report.framework_hints)

    print_section("Notable deployment signals")
    if report.deployment_signals:
        for signal in report.deployment_signals:
            print(f"- {signal}")
    else:
        print("No memory banks, large tensor entries, thresholds, or backbone hints detected.")

    print_section("Possible task hints")
    if report.task_hints:
        for hint in report.task_hints:
            examples = ", ".join(hint["matches"][:8])
            print(f"- {hint['task']}: {hint['confidence']} confidence; matched {examples}")
    else:
        print("No task-specific key name hints detected.")

    print_section("Conversion note")
    print(
        "Task hints are heuristic only. Exact model conversion still requires "
        "the model architecture, checkpoint loading rule, input shape, output "
        "spec, and preprocessing/postprocessing rules."
    )


def _state_dict_preview(
    state_dict: Mapping[Any, Any], max_items: int
) -> list[dict[str, Any]]:
    preview = []
    for index, (key, value) in enumerate(state_dict.items()):
        if index >= max_items:
            break
        preview.append(
            {
                "key": str(key),
                "tensor_like": is_tensor_like(value),
                "shape": safe_shape(value),
                "dtype": safe_dtype(value),
            }
        )
    return preview


def detect_checkpoint_kind(obj: Any, path: Path | None = None) -> str:
    if path and is_torchscript_archive(path):
        return "TorchScript archive"

    if _is_nn_module(obj):
        return "full PyTorch model object"

    if is_state_dict_like(obj):
        return "raw state_dict"

    if isinstance(obj, Mapping):
        if is_lightning_checkpoint(obj):
            return "PyTorch Lightning checkpoint"
        if any(key in obj for key in STATE_DICT_KEYS) or "optimizer" in obj:
            return "checkpoint dict"
        return "generic dict"

    return "unknown"


def is_torchscript_archive(path: Path) -> bool:
    if not path.is_file() or not zipfile.is_zipfile(path):
        return False

    try:
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
    except Exception:
        return False

    normalized = [name.split("/", 1)[-1] if "/" in name else name for name in names]
    has_code = any("/code/" in f"/{name}" or name.startswith("code/") for name in names)
    has_constants = any(name.endswith("constants.pkl") for name in normalized)
    has_data = any(name.endswith("data.pkl") for name in normalized)
    return has_code and has_constants and has_data


def is_state_dict_like(obj: Any) -> bool:
    if not isinstance(obj, Mapping) or not obj:
        return False

    string_keys = sum(1 for key in obj if isinstance(key, str))
    tensor_values = sum(1 for value in obj.values() if is_tensor_like(value))
    return string_keys == len(obj) and tensor_values >= max(1, int(len(obj) * 0.6))


def is_lightning_checkpoint(obj: Mapping[Any, Any]) -> bool:
    lightning_keys = {
        "pytorch-lightning_version",
        "hyper_parameters",
        "callbacks",
        "optimizer_states",
        "lr_schedulers",
    }
    return "state_dict" in obj and any(key in obj for key in lightning_keys)


def find_state_dict(obj: Any) -> tuple[str, Mapping[Any, Any] | None]:
    if is_state_dict_like(obj):
        return "top-level object", obj

    if not isinstance(obj, Mapping):
        return "", None

    for key in STATE_DICT_KEYS:
        value = obj.get(key)
        if is_state_dict_like(value):
            return key, value

    for key in ("model", "module", "net", "network"):
        value = obj.get(key)
        if is_state_dict_like(value):
            return key, value

    return "", None


def print_state_dict(name: str, state_dict: Mapping[Any, Any], max_items: int) -> None:
    print_section("State dict")
    print(f"Source: {name}")
    print(f"Entries: {len(state_dict)}")
    print(f"Showing first {min(max_items, len(state_dict))} entries")

    for index, (key, value) in enumerate(state_dict.items()):
        if index >= max_items:
            break
        tensor_like = is_tensor_like(value)
        print(
            f"- {key}: tensor_like={tensor_like}, "
            f"shape={safe_shape(value)}, dtype={safe_dtype(value)}"
        )


def print_metadata(obj: Any) -> None:
    if not isinstance(obj, Mapping):
        return

    present = [key for key in METADATA_KEYS if key in obj]
    if not present:
        return

    print_section("Common metadata")
    for key in present:
        print(f"{key}: {truncate_repr(obj[key])}")


def summarize_tensors(
    checkpoint: Any,
    *,
    state_dict: Mapping[Any, Any] | None = None,
    max_items: int = 40,
) -> TensorSummary:
    candidates: list[tuple[str, Any]] = []
    if state_dict is not None:
        candidates.extend((str(key), value) for key, value in state_dict.items())
    candidates.extend(
        (item.path, item.value)
        for item in collect_mapping_entries(checkpoint)
        if is_tensor_like(item.value)
    )

    seen: set[int] = set()
    summary = TensorSummary()
    largest: list[dict[str, Any]] = []
    for name, value in candidates:
        value_id = id(value)
        if value_id in seen:
            continue
        seen.add(value_id)
        if not is_tensor_like(value):
            continue
        summary.tensor_count += 1
        numel = safe_tensor_numel(value)
        nbytes = safe_tensor_nbytes(value)
        dtype = safe_dtype(value)
        if numel is not None:
            summary.parameter_count += numel
        if nbytes is not None:
            summary.estimated_bytes += nbytes
        summary.dtype_histogram[dtype] = summary.dtype_histogram.get(dtype, 0) + 1
        largest.append(
            {
                "name": name,
                "shape": safe_shape(value),
                "dtype": dtype,
                "numel": numel,
                "estimated_bytes": nbytes,
                "estimated_size": format_file_size(nbytes) if nbytes is not None else "-",
            }
        )

    largest.sort(key=lambda item: item["estimated_bytes"] or 0, reverse=True)
    summary.largest_tensors = largest[:max_items]
    return summary


def safe_tensor_numel(obj: Any) -> int | None:
    if not is_tensor_like(obj):
        return None
    try:
        numel = obj.numel() if callable(getattr(obj, "numel", None)) else None
        if isinstance(numel, int):
            return numel
    except Exception:
        return None
    try:
        shape = getattr(obj, "shape")
        total = 1
        for dim in shape:
            if not isinstance(dim, int):
                return None
            total *= dim
        return total
    except Exception:
        return None


def summarize_nested(checkpoint: Any, max_items: int = 40) -> list[NestedSummary]:
    if not isinstance(checkpoint, Mapping):
        return []
    summaries: list[NestedSummary] = []
    for key in NESTED_INSPECTION_KEYS:
        if key not in checkpoint:
            continue
        value = checkpoint[key]
        mapping_keys: list[str] = []
        mapping_key_count = None
        if isinstance(value, Mapping):
            mapping_keys = [str(item) for item in list(value.keys())[:max_items]]
            mapping_key_count = len(value)
        state_like = is_state_dict_like(value)
        summaries.append(
            NestedSummary(
                key=key,
                type_name=_type_name(value),
                mapping_keys=mapping_keys,
                mapping_key_count=mapping_key_count,
                state_dict_like=state_like,
                tensor_summary=summarize_tensors(value, max_items=max_items)
                if isinstance(value, Mapping)
                else None,
            )
        )
    return summaries


def print_tensor_summary(summary: TensorSummary) -> None:
    print_section("Tensor summary")
    print(f"Tensor count: {summary.tensor_count}")
    print(f"Parameter count: {summary.parameter_count}")
    print(f"Estimated tensor bytes: {format_file_size(summary.estimated_bytes)}")
    print("Dtype histogram:")
    if summary.dtype_histogram:
        for dtype, count in sorted(summary.dtype_histogram.items()):
            print(f"- {dtype}: {count}")
    else:
        print("- None")
    print("Largest tensors:")
    if summary.largest_tensors:
        for item in summary.largest_tensors[:10]:
            print(
                f"- {item['name']}: shape={item['shape']}, dtype={item['dtype']}, "
                f"numel={item['numel']}, estimated_size={item['estimated_size']}"
            )
    else:
        print("- None")


def print_nested_summary(values: list[NestedSummary], max_items: int) -> None:
    print_section("Nested inspection")
    if not values:
        print("No known nested checkpoint sections found.")
        return
    for item in values:
        print(f"- {item.key}: type={item.type_name}")
        if item.mapping_key_count is not None:
            shown = min(max_items, len(item.mapping_keys))
            print(f"  keys: showing {shown} of {item.mapping_key_count}")
            if item.mapping_keys:
                print(f"  first keys: {', '.join(item.mapping_keys[:max_items])}")
        print(f"  state_dict_like: {'yes' if item.state_dict_like else 'no'}")
        if item.tensor_summary and item.tensor_summary.tensor_count:
            print(
                "  tensors: "
                f"count={item.tensor_summary.tensor_count}, "
                f"params={item.tensor_summary.parameter_count}, "
                f"bytes={format_file_size(item.tensor_summary.estimated_bytes)}"
            )


def collect_mapping_entries(
    obj: Any, max_depth: int = 4, max_keys: int = 5000
) -> list["MappingEntry"]:
    entries: list[MappingEntry] = []
    seen: set[int] = set()

    def visit(value: Any, depth: int, prefix: str = "") -> None:
        if len(entries) >= max_keys or depth > max_depth:
            return
        if id(value) in seen:
            return
        seen.add(id(value))

        if isinstance(value, Mapping):
            for key, child in value.items():
                if len(entries) >= max_keys:
                    return
                key_text = str(key)
                path = f"{prefix}.{key_text}" if prefix else key_text
                entries.append(MappingEntry(path=path, key=key_text, value=child))
                if isinstance(child, Mapping):
                    visit(child, depth + 1, path)
                elif isinstance(child, (list, tuple)) and depth + 1 <= max_depth:
                    for index, item in enumerate(child[:25]):
                        visit(item, depth + 1, f"{path}[{index}]")

    visit(obj, 0)
    return entries


class MappingEntry:
    def __init__(self, path: str, key: str, value: Any) -> None:
        self.path = path
        self.key = key
        self.value = value


def print_deployment_signals(
    checkpoint: Any, state_dict: Mapping[Any, Any] | None
) -> None:
    print_section("Notable deployment signals")

    entries = collect_mapping_entries(checkpoint)
    if state_dict is not None:
        entries.extend(
            MappingEntry(path=str(key), key=str(key), value=value)
            for key, value in state_dict.items()
        )

    memory_bank_entries = [
        item for item in entries if "memory_bank" in item.key.lower()
    ]
    threshold_entries = [
        item for item in entries if _matches_any(item.key.lower(), THRESHOLD_KEYWORDS)
    ]
    large_tensor_entries = []
    for item in entries:
        nbytes = safe_tensor_nbytes(item.value)
        if nbytes is not None and nbytes >= LARGE_TENSOR_BYTES:
            large_tensor_entries.append((item, nbytes))

    hyper_parameters = checkpoint.get("hyper_parameters") if isinstance(checkpoint, Mapping) else None
    backbone_hints = find_backbone_layer_hints(hyper_parameters)

    printed = False
    for item in memory_bank_entries:
        print(
            f"- memory_bank: {item.path}, shape={safe_shape(item.value)}, "
            f"dtype={safe_dtype(item.value)}"
        )
        printed = True

    for item, nbytes in sorted(large_tensor_entries, key=lambda pair: pair[1], reverse=True)[:10]:
        print(
            f"- large tensor: {item.path}, shape={safe_shape(item.value)}, "
            f"dtype={safe_dtype(item.value)}, estimated_size={format_file_size(nbytes)}"
        )
        printed = True

    if threshold_entries:
        examples = ", ".join(item.path for item in threshold_entries[:20])
        print(f"- threshold-related keys: {examples}")
        printed = True

    if backbone_hints:
        for key, value in backbone_hints[:12]:
            print(f"- backbone/layer hint: {key}={truncate_repr(value, max_length=160)}")
        printed = True

    if not printed:
        print("No memory banks, large tensor entries, thresholds, or backbone hints detected.")


def deployment_signals(
    checkpoint: Any, state_dict: Mapping[Any, Any] | None
) -> list[str]:
    entries = collect_mapping_entries(checkpoint)
    if state_dict is not None:
        entries.extend(
            MappingEntry(path=str(key), key=str(key), value=value)
            for key, value in state_dict.items()
        )

    signals: list[str] = []
    for item in entries:
        if "memory_bank" in item.key.lower():
            signals.append(
                f"memory_bank: {item.path}, shape={safe_shape(item.value)}, "
                f"dtype={safe_dtype(item.value)}"
            )

    large_tensor_entries = []
    for item in entries:
        nbytes = safe_tensor_nbytes(item.value)
        if nbytes is not None and nbytes >= LARGE_TENSOR_BYTES:
            large_tensor_entries.append((item, nbytes))
    for item, nbytes in sorted(large_tensor_entries, key=lambda pair: pair[1], reverse=True)[:10]:
        signals.append(
            f"large tensor: {item.path}, shape={safe_shape(item.value)}, "
            f"dtype={safe_dtype(item.value)}, estimated_size={format_file_size(nbytes)}"
        )

    threshold_entries = [
        item for item in entries if _matches_any(item.key.lower(), THRESHOLD_KEYWORDS)
    ]
    if threshold_entries:
        signals.append(
            "threshold-related keys: "
            + ", ".join(item.path for item in threshold_entries[:20])
        )

    hyper_parameters = checkpoint.get("hyper_parameters") if isinstance(checkpoint, Mapping) else None
    for key, value in find_backbone_layer_hints(hyper_parameters)[:12]:
        signals.append(f"backbone/layer hint: {key}={truncate_repr(value, max_length=160)}")
    return signals


def detect_framework_hints(checkpoint: Any, key_names: list[str] | None = None) -> list[FrameworkHint]:
    evidence_by_framework: dict[str, list[str]] = {}

    def add(framework: str, evidence: str) -> None:
        evidence_by_framework.setdefault(framework, [])
        if evidence not in evidence_by_framework[framework]:
            evidence_by_framework[framework].append(evidence)

    type_names = collect_type_names(checkpoint)
    for type_name in type_names:
        lower = type_name.lower()
        if "ultralytics" in lower:
            add("ultralytics", f"object type contains '{type_name}'")
        if "anomalib" in lower:
            add("anomalib", f"object type contains '{type_name}'")

    blob = " ".join((key_names or [])[:5000]).lower()
    if "ultralytics" in blob or "yolo" in blob:
        add("ultralytics", "checkpoint keys contain Ultralytics/YOLO hints")
    if "anomalib" in blob or "patchcore" in blob or "memory_bank" in blob:
        add("anomalib", "checkpoint keys contain Anomalib/PatchCore hints")

    hints: list[FrameworkHint] = []
    for framework, evidence in sorted(evidence_by_framework.items()):
        confidence = "high" if any("object type" in item for item in evidence) else "medium"
        hints.append(FrameworkHint(framework=framework, confidence=confidence, evidence=evidence))
    return hints


def collect_type_names(obj: Any, max_depth: int = 3, max_items: int = 1000) -> list[str]:
    names: list[str] = []
    seen: set[int] = set()

    def visit(value: Any, depth: int) -> None:
        if len(names) >= max_items or depth > max_depth:
            return
        value_id = id(value)
        if value_id in seen:
            return
        seen.add(value_id)
        type_name = _type_name(value)
        if type_name not in names:
            names.append(type_name)
        if isinstance(value, Mapping):
            for child in value.values():
                visit(child, depth + 1)
        elif isinstance(value, (list, tuple)):
            for child in value[:25]:
                visit(child, depth + 1)

    visit(obj, 0)
    return names


def framework_from_unsupported_global(text: str) -> FrameworkHint | None:
    lower = text.lower()
    if "ultralytics." in lower:
        return FrameworkHint(
            framework="ultralytics",
            confidence="high",
            evidence=["safe-load error references an Ultralytics global"],
        )
    if "anomalib." in lower:
        return FrameworkHint(
            framework="anomalib",
            confidence="high",
            evidence=["safe-load error references an Anomalib global"],
        )
    return None


def print_framework_hints(hints: list[FrameworkHint]) -> None:
    print_section("Framework hints")
    if not hints:
        print("No framework-specific hints detected.")
        return
    for hint in hints:
        print(f"- {hint.framework}: {hint.confidence} confidence")
        for evidence in hint.evidence:
            print(f"  evidence: {evidence}")


def find_backbone_layer_hints(value: Any, prefix: str = "") -> list[tuple[str, Any]]:
    hints: list[tuple[str, Any]] = []
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            if _matches_any(key_text.lower(), BACKBONE_LAYER_KEYWORDS):
                hints.append((path, child))
            if isinstance(child, Mapping):
                hints.extend(find_backbone_layer_hints(child, path))
    return hints


def print_task_hints(key_names: list[str]) -> None:
    print_section("Possible task hints")
    hints = detect_task_hints(key_names)
    if not hints:
        print("No task-specific key name hints detected.")
        return

    for task_name, confidence, matches in hints:
        examples = ", ".join(matches[:8])
        print(f"- {task_name}: {confidence} confidence; matched {examples}")


def detect_task_hints(key_names: list[str]) -> list[tuple[str, str, list[str]]]:
    lowered = [(key, key.lower()) for key in key_names]
    hints: list[tuple[str, str, list[str]]] = []

    for task_name, keywords in TASK_KEYWORDS.items():
        matches: list[str] = []
        matched_keywords: set[str] = set()
        for original, lower in lowered:
            for keyword in keywords:
                if keyword in lower:
                    matched_keywords.add(keyword)
                    if original not in matches:
                        matches.append(original)
                    break

        if not matches:
            continue
        if task_name == "anomaly_detection" and matched_keywords == {"feature_extractor"}:
            continue
        if task_name == "embedding_or_metric_learning" and matched_keywords == {"metric"}:
            metric_only_matches = [match for match in matches if "train_metric" in match.lower()]
            if len(metric_only_matches) == len(matches):
                continue

        confidence = _confidence_level(task_name, matches, matched_keywords)
        hints.append((task_name, confidence, matches))

    confidence_rank = {"high": 0, "medium": 1, "low": 2}
    task_rank = {
        "segmentation": 0,
        "object_detection": 1,
        "anomaly_detection": 2,
        "classification": 3,
        "pose_estimation": 4,
        "embedding_or_metric_learning": 8,
    }
    hints.sort(key=lambda item: (confidence_rank[item[1]], task_rank.get(item[0], 5), item[0]))
    return hints


def _confidence_level(
    task_name: str, matches: list[str], matched_keywords: set[str]
) -> str:
    match_count = len(matches)
    keyword_count = len(matched_keywords)

    if task_name == "anomaly_detection":
        matched_lower = {match.lower() for match in matches}
        has_memory_bank = any("memory_bank" in key for key in matched_lower)
        has_feature_extractor = any("feature_extractor" in key for key in matched_lower)
        if has_memory_bank and has_feature_extractor:
            return "high"
        if ANOMALY_STRONG_KEYS.intersection(matched_keywords):
            return "medium" if match_count < 5 and keyword_count < 3 else "high"

    if match_count >= 5 or keyword_count >= 3:
        return "high"
    if match_count >= 2 or keyword_count >= 2:
        return "medium"
    return "low"


def _matches_any(value: str, keywords: tuple[str, ...] | set[str]) -> bool:
    return any(keyword in value for keyword in keywords)


def _is_nn_module(obj: Any) -> bool:
    try:
        import torch.nn as nn
    except Exception:
        return False
    return isinstance(obj, nn.Module)


def _type_name(obj: Any) -> str:
    obj_type = type(obj)
    return f"{obj_type.__module__}.{obj_type.__qualname__}"
