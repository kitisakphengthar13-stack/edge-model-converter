from __future__ import annotations

import argparse
import inspect
import json
import sys
from pathlib import Path
from typing import Any

from . import __version__
from .checkpoint import (
    load_checkpoint_safe,
    load_checkpoint_unsafe,
    summarize_loaded_checkpoint,
)
from .compare import compare_onnx, render_compare_text
from .dry_run import run_model_dry_run
from .export_assessment import (
    assess_export_from_model_path,
    assess_export_from_spec,
)
from .inspect_pt import build_inspection_report, inspect_checkpoint, print_inspection_report
from .load_plan import print_loading_plan
from .onnx_export import export_onnx_from_spec
from .onnx_validate import (
    build_onnx_inspection_report,
    parse_cli_shape,
    print_onnx_report,
    validate_onnx_file,
)
from .readiness import SUPPORTED_READINESS_PROFILES
from .reports import (
    envelope_from_onnx_report,
    envelope_from_pytorch_report,
    messages_reach_threshold,
    render_markdown_envelope,
    render_scan_markdown,
)
from .scan import scan_artifacts
from .spec import SpecValidationResult, print_validation_result, validate_spec_file
from .tensorrt_plan import create_tensorrt_plan
from .utils import print_section


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="edge-inspect",
        description=(
            "Inspect PyTorch checkpoints and ONNX artifacts for edge deployment readiness."
        ),
        epilog="Backward compatible module entrypoint: python -m converter.cli",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"edge-inspect {__version__}",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser(
        "inspect",
        help="Inspect a .pt, .pth, or .ckpt file without exporting it.",
    )
    inspect_parser.add_argument("path", help="Path to the PyTorch checkpoint file.")
    inspect_parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of state_dict entries to print. Default: 40.",
    )
    inspect_parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow full pickle loading for trusted local files only.",
    )
    inspect_parser.add_argument(
        "--json",
        dest="json_output",
        help="Optional path to write a structured JSON inspection report.",
    )
    inspect_parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format. Default: text.",
    )
    inspect_parser.add_argument("--output", help="Optional report output path.")
    inspect_parser.set_defaults(func=inspect_command)

    validate_spec_parser = subparsers.add_parser(
        "validate-spec",
        help="Validate a generic model conversion spec.yaml file.",
    )
    validate_spec_parser.add_argument("path", help="Path to the YAML spec file.")
    validate_spec_parser.set_defaults(func=validate_spec_command)

    assess_export_parser = subparsers.add_parser(
        "assess-export",
        help=(
            "Assess the recommended ONNX export route without running exporters."
        ),
    )
    assess_export_parser.add_argument(
        "path", help="Path to a spec YAML file or PyTorch checkpoint/model file."
    )
    assess_export_parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow unsafe pickle loading for checkpoint-path assessment only.",
    )
    assess_export_parser.add_argument(
        "--max-items",
        type=int,
        default=80,
        help="Maximum checkpoint hint items for model-path assessment. Default: 80.",
    )
    assess_export_parser.set_defaults(func=assess_export_command)

    plan_load_parser = subparsers.add_parser(
        "plan-load",
        help="Plan future model loading from a spec without executing it.",
    )
    plan_load_parser.add_argument("path", help="Path to the YAML spec file.")
    plan_load_parser.set_defaults(func=plan_load_command)

    check_checkpoint_parser = subparsers.add_parser(
        "check-checkpoint",
        help="Load only the checkpoint declared by a valid spec and summarize it.",
    )
    check_checkpoint_parser.add_argument("path", help="Path to the YAML spec file.")
    check_checkpoint_parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of state_dict entries to print. Default: 40.",
    )
    check_checkpoint_parser.set_defaults(func=check_checkpoint_command)

    dry_run_parser = subparsers.add_parser(
        "dry-run-model",
        help="Instantiate a trusted local model and run a CPU dummy forward pass.",
    )
    dry_run_parser.add_argument("path", help="Path to the YAML spec file.")
    dry_run_parser.add_argument(
        "--allow-imports",
        action="store_true",
        help="Allow importing and instantiating the model module declared in the spec.",
    )
    dry_run_parser.add_argument(
        "--max-items",
        type=int,
        default=40,
        help="Maximum number of output or load diagnostic entries to print. Default: 40.",
    )
    strict_group = dry_run_parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=None,
        help="Force strict state_dict loading.",
    )
    strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow missing or unexpected state_dict keys.",
    )
    dry_run_parser.add_argument(
        "--prefix-to-strip",
        help="Override checkpoint.prefix_to_strip before loading state_dict.",
    )
    dry_run_parser.add_argument(
        "--device",
        default="cpu",
        help="Dry-run device. Phase 4 supports only cpu.",
    )
    dry_run_parser.set_defaults(func=dry_run_model_command)

    export_parser = subparsers.add_parser(
        "export-onnx",
        help="Run the guarded generic PyTorch-to-ONNX fallback exporter from a spec.",
    )
    export_parser.add_argument("path", help="Path to the YAML spec file.")
    export_parser.add_argument(
        "--allow-imports",
        action="store_true",
        help="Allow importing and instantiating the model module declared in the spec.",
    )
    export_parser.add_argument("--output", help="Output ONNX path.")
    export_parser.add_argument("--opset", type=int, help="Override conversion.opset.")
    export_strict_group = export_parser.add_mutually_exclusive_group()
    export_strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        default=None,
        help="Force strict state_dict loading.",
    )
    export_strict_group.add_argument(
        "--no-strict",
        dest="strict",
        action="store_false",
        help="Allow missing or unexpected state_dict keys.",
    )
    export_parser.add_argument(
        "--prefix-to-strip",
        help="Override checkpoint.prefix_to_strip before loading state_dict.",
    )
    export_parser.add_argument(
        "--device",
        default="cpu",
        help="Export device. Phase 5 supports only cpu.",
    )
    export_parser.add_argument(
        "--no-dynamo",
        action="store_true",
        help="Use the legacy torch.onnx exporter directly.",
    )
    export_parser.add_argument(
        "--no-fallback-legacy",
        action="store_true",
        help="Do not retry legacy export if dynamo export fails.",
    )
    export_parser.set_defaults(func=export_onnx_command)

    validate_onnx_parser = subparsers.add_parser(
        "validate-onnx",
        help="Validate an ONNX artifact and run ONNX Runtime CPU inference.",
    )
    validate_onnx_parser.add_argument("path", help="Path to the ONNX file.")
    validate_onnx_parser.add_argument("--spec", help="Optional spec.yaml path.")
    validate_onnx_parser.add_argument(
        "--input-shape",
        help="Comma-separated concrete input shape, for example 1,3,224,224.",
    )
    validate_onnx_parser.add_argument("--input-dtype", help="Override input dtype.")
    validate_onnx_parser.add_argument("--input-name", help="Override input name.")
    validate_onnx_parser.add_argument(
        "--target",
        choices=SUPPORTED_READINESS_PROFILES,
        default="generic",
        help="Edge readiness target profile. Default: generic.",
    )
    validate_onnx_parser.add_argument(
        "--max-items",
        type=int,
        default=20,
        help="Maximum number of inputs, outputs, or summaries to print. Default: 20.",
    )
    validate_onnx_parser.add_argument(
        "--json",
        dest="json_output",
        help="Optional path to write a structured JSON ONNX inspection report.",
    )
    validate_onnx_parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format. Default: text.",
    )
    validate_onnx_parser.add_argument("--output", help="Optional report output path.")
    validate_onnx_parser.add_argument(
        "--fail-on",
        choices=("warning", "error"),
        help="Return nonzero if the report contains this severity or higher.",
    )
    validate_onnx_parser.set_defaults(func=validate_onnx_command)

    scan_parser = subparsers.add_parser(
        "scan",
        help="Batch inspect supported PyTorch checkpoint and ONNX artifacts.",
    )
    scan_parser.add_argument("path", help="Directory, file, or glob to scan.")
    scan_parser.add_argument(
        "--include",
        action="append",
        help="Include pattern for directory scans. Can be repeated.",
    )
    scan_parser.add_argument(
        "--unsafe-load",
        action="store_true",
        help="Allow unsafe trusted-local PyTorch checkpoint loading during scan.",
    )
    scan_parser.add_argument("--input-shape", help="Comma-separated ONNX dummy input shape.")
    scan_parser.add_argument(
        "--target",
        choices=SUPPORTED_READINESS_PROFILES,
        default="generic",
        help="Edge readiness target profile for ONNX artifacts. Default: generic.",
    )
    scan_parser.add_argument(
        "--format",
        choices=("text", "json", "markdown"),
        default="text",
        help="Output format. Default: text.",
    )
    scan_parser.add_argument("--output", help="Optional aggregate report output path.")
    scan_parser.add_argument(
        "--fail-on",
        choices=("warning", "error"),
        help="Return nonzero if any scanned report contains this severity or higher.",
    )
    scan_parser.add_argument("--max-items", type=int, default=20)
    scan_parser.set_defaults(func=scan_command)

    compare_parser = subparsers.add_parser(
        "compare",
        help="Compare two artifacts. ONNX-vs-ONNX is supported first.",
    )
    compare_parser.add_argument("left")
    compare_parser.add_argument("right")
    compare_parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format. Default: text.",
    )
    compare_parser.add_argument("--output", help="Optional comparison output path.")
    compare_parser.set_defaults(func=compare_command)

    plan_tensorrt_parser = subparsers.add_parser(
        "plan-tensorrt",
        help="Generate an optional downstream TensorRT/trtexec plan without running TensorRT.",
    )
    plan_tensorrt_parser.add_argument("path", help="Path to the ONNX file.")
    plan_tensorrt_parser.add_argument("--spec", help="Optional spec.yaml path.")
    plan_tensorrt_parser.add_argument(
        "--target",
        default="generic",
        help="Target device/runtime label. Default: generic.",
    )
    plan_tensorrt_parser.add_argument(
        "--precision",
        default="fp16",
        help="TensorRT precision: fp32, fp16, or int8. Default: fp16.",
    )
    plan_tensorrt_parser.add_argument("--engine-output", help="Planned engine path.")
    plan_tensorrt_parser.add_argument(
        "--workspace-mb",
        type=int,
        help="TensorRT workspace size in MB.",
    )
    plan_tensorrt_parser.add_argument("--input-name", help="Input name for dynamic shapes.")
    plan_tensorrt_parser.add_argument("--min-shape", help="Dynamic min shape, e.g. 1x3x224x224.")
    plan_tensorrt_parser.add_argument("--opt-shape", help="Dynamic opt shape, e.g. 1x3x224x224.")
    plan_tensorrt_parser.add_argument("--max-shape", help="Dynamic max shape, e.g. 4x3x224x224.")
    plan_tensorrt_parser.add_argument("--timing-cache", help="TensorRT timing cache file path.")
    plan_tensorrt_parser.add_argument(
        "--verbose-trtexec",
        action="store_true",
        help="Include --verbose in the generated trtexec command.",
    )
    plan_tensorrt_parser.set_defaults(func=plan_tensorrt_command)

    return parser


def inspect_command(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"Error: file not found: {path}", file=sys.stderr)
        return 1

    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    try:
        checkpoint = load_checkpoint(path, unsafe_load=args.unsafe_load)
    except Exception as exc:
        if args.unsafe_load:
            print(f"Error: failed to load checkpoint: {exc}", file=sys.stderr)
        else:
            print(f"Error: safe checkpoint loading failed: {exc}", file=sys.stderr)
            print(
                "This file may require full pickle loading. Re-run with "
                "--unsafe-load only if the file is trusted and local.",
                file=sys.stderr,
            )
        return 2

    report = build_inspection_report(path, checkpoint, max_items=args.max_items)
    envelope = envelope_from_pytorch_report(report.to_dict())
    if args.format == "markdown":
        rendered = render_markdown_envelope(envelope)
        _emit_or_write(rendered, args.output)
    elif args.format == "json":
        rendered = json.dumps(envelope.to_dict(), indent=2) + "\n"
        _emit_or_write(rendered, args.output)
    else:
        print_inspection_report(report, max_items=args.max_items)
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(report.to_dict(), file, indent=2)
            file.write("\n")
    if args.output and args.format == "text":
        rendered = render_markdown_envelope(envelope) if args.output.lower().endswith(".md") else json.dumps(envelope.to_dict(), indent=2) + "\n"
        _emit_or_write(rendered, args.output)
    return 0


def validate_spec_command(args: argparse.Namespace) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"Error: spec file not found: {path}", file=sys.stderr)
        return 1

    result = validate_spec_file(path)
    print_validation_result(result)
    return 0 if result.valid else 2


def assess_export_command(args: argparse.Namespace) -> int:
    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    path = Path(args.path)
    try:
        if path.suffix.lower() in {".yaml", ".yml"}:
            result = assess_export_from_spec(str(path))
        else:
            result = assess_export_from_model_path(
                str(path),
                unsafe_load=args.unsafe_load,
                max_items=args.max_items,
            )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print_export_assessment(result)
    return 0


def plan_load_command(args: argparse.Namespace) -> int:
    result = _validate_spec_or_report(Path(args.path))
    if result is None:
        return 2

    print_loading_plan(result.spec)
    _print_spec_warnings(result)
    return 0


def check_checkpoint_command(args: argparse.Namespace) -> int:
    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    result = _validate_spec_or_report(Path(args.path))
    if result is None:
        return 2
    spec = result.spec

    checkpoint_spec = spec["checkpoint"]
    load_mode = checkpoint_spec["load_mode"]
    state_dict_key = checkpoint_spec.get("state_dict_key")

    if checkpoint_spec.get("kind") == "none" and load_mode == "none":
        print("Checkpoint: none declared; no checkpoint was loaded.")
        return 0

    checkpoint_path_value = spec.get("checkpoint_path")
    if not checkpoint_path_value:
        print("Error: checkpoint_path is required for check-checkpoint", file=sys.stderr)
        return 1

    checkpoint_path = Path(checkpoint_path_value)
    if not checkpoint_path.is_file():
        print(f"Error: checkpoint file not found: {checkpoint_path}", file=sys.stderr)
        return 1

    if load_mode in {"custom_loader", "external_loader"}:
        print(
            f"Checkpoint load mode is '{load_mode}'. Custom or external loader "
            "execution is not implemented in Phase 3, so no checkpoint was loaded."
        )
        return 0

    try:
        if load_mode == "safe_weights_only":
            load_result = load_checkpoint_safe(checkpoint_path)
        elif load_mode == "unsafe_trusted_local":
            load_result = load_checkpoint_unsafe(checkpoint_path)
        else:
            print(
                f"Checkpoint load mode '{load_mode}' is custom/unknown. "
                "Checkpoint loading is not implemented for this mode in Phase 3."
            )
            return 0
    except Exception as exc:
        print(f"Error: failed to load checkpoint: {exc}", file=sys.stderr)
        return 2

    for warning in load_result.warnings:
        print(f"Warning: {warning}", file=sys.stderr)

    summarize_loaded_checkpoint(
        load_result.checkpoint,
        state_dict_key=state_dict_key,
        max_items=args.max_items,
    )
    return 0


def dry_run_model_command(args: argparse.Namespace) -> int:
    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    result = _validate_spec_or_report(Path(args.path))
    if result is None:
        return 2

    return run_model_dry_run(
        result.spec,
        allow_imports=args.allow_imports,
        max_items=args.max_items,
        strict_override=args.strict,
        prefix_to_strip=args.prefix_to_strip,
        device=args.device,
    )


def export_onnx_command(args: argparse.Namespace) -> int:
    try:
        result = export_onnx_from_spec(
            str(args.path),
            output_path=args.output,
            allow_imports=args.allow_imports,
            opset=args.opset,
            strict=args.strict,
            prefix_to_strip=args.prefix_to_strip,
            device=args.device,
            dynamo=not args.no_dynamo,
            fallback_legacy=not args.no_fallback_legacy,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print_section("Export result")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


def validate_onnx_command(args: argparse.Namespace) -> int:
    if args.max_items < 1:
        print("Error: --max-items must be at least 1", file=sys.stderr)
        return 1

    try:
        shape = parse_cli_shape(args.input_shape) if args.input_shape else None
        report = build_onnx_inspection_report(
            args.path,
            spec_path=args.spec,
            input_shape=shape,
            input_dtype=args.input_dtype,
            input_name=args.input_name,
            max_items=args.max_items,
            target=args.target,
        )
        result = report.to_dict() | {
            "onnx_path": report.path,
            "checker_passed": report.validation.checker_passed,
            "ort_session_created": report.validation.ort_session_created,
            "inference_passed": report.validation.inference_passed,
            "input_names": [item.name for item in report.inputs],
            "output_names": [item.name for item in report.outputs],
            "success": report.validation.checker_passed and report.validation.ort_session_created,
        }
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    envelope = envelope_from_onnx_report(result)
    if args.format == "markdown":
        _emit_or_write(render_markdown_envelope(envelope), args.output)
    elif args.format == "json":
        _emit_or_write(json.dumps(envelope.to_dict(), indent=2) + "\n", args.output)
    else:
        print_onnx_report(report, max_items=args.max_items)
        print_section("Validation result")
        for key in (
            "onnx_path",
            "checker_passed",
            "ort_session_created",
            "inference_passed",
            "success",
        ):
            print(f"{key}: {result.get(key)}")
    if args.json_output:
        output_path = Path(args.json_output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as file:
            json.dump(result, file, indent=2)
            file.write("\n")
        print(f"json_report: {output_path}")
    if args.output and args.format == "text":
        _emit_or_write(json.dumps(envelope.to_dict(), indent=2) + "\n", args.output)
    return 3 if messages_reach_threshold(envelope.warnings, envelope.errors, args.fail_on) else 0


def scan_command(args: argparse.Namespace) -> int:
    try:
        shape = parse_cli_shape(args.input_shape) if args.input_shape else None
        envelopes = scan_artifacts(
            args.path,
            includes=args.include,
            unsafe_load=args.unsafe_load,
            input_shape=shape,
            max_items=args.max_items,
            target=args.target,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    payload = {"artifacts": [item.to_dict() for item in envelopes]}
    if args.format == "json":
        rendered = json.dumps(payload, indent=2) + "\n"
    elif args.format == "markdown":
        rendered = render_scan_markdown(envelopes)
    else:
        rendered = render_scan_markdown(envelopes)
    _emit_or_write(rendered, args.output)
    triggered = any(
        messages_reach_threshold(item.warnings, item.errors, args.fail_on)
        for item in envelopes
    )
    return 3 if triggered else 0


def compare_command(args: argparse.Namespace) -> int:
    left = Path(args.left)
    right = Path(args.right)
    if left.suffix.lower() != ".onnx" or right.suffix.lower() != ".onnx":
        print("Error: compare currently supports ONNX-vs-ONNX only.", file=sys.stderr)
        return 2
    try:
        result = compare_onnx(str(left), str(right))
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    rendered = (
        json.dumps(result, indent=2) + "\n"
        if args.format == "json"
        else render_compare_text(result)
    )
    _emit_or_write(rendered, args.output)
    return 0


def _emit_or_write(text: str, output: str | None) -> None:
    if output:
        output_path = Path(output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")
    else:
        print(text, end="")


def plan_tensorrt_command(args: argparse.Namespace) -> int:
    try:
        result = create_tensorrt_plan(
            args.path,
            spec_path=args.spec,
            target=args.target,
            precision=args.precision,
            engine_output=args.engine_output,
            workspace_mb=args.workspace_mb,
            min_shape=args.min_shape,
            opt_shape=args.opt_shape,
            max_shape=args.max_shape,
            input_name=args.input_name,
            timing_cache=args.timing_cache,
            verbose=args.verbose_trtexec,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2

    print_section("Plan result")
    for key, value in result.items():
        print(f"{key}: {value}")
    return 0


def _validate_spec_or_report(path: Path) -> SpecValidationResult | None:
    if not path.is_file():
        print(f"Error: spec file not found: {path}", file=sys.stderr)
        return None

    result = validate_spec_file(path)
    if not result.valid:
        print_validation_result(result)
        return None
    if result.spec is None:
        print("Error: spec did not load into a dictionary", file=sys.stderr)
        return None
    return result


def _print_spec_warnings(result: SpecValidationResult) -> None:
    if not result.warnings:
        return
    print_section("Spec warnings")
    for warning in result.warnings:
        print(f"- {warning}")


def print_export_assessment(result: dict[str, Any]) -> None:
    detected = result["detected_source"]
    official = result["official_source_exporter"]
    toolkit = result["toolkit_generic_exporter"]
    recommendation = result["recommended_route"]

    print_section("Export capability assessment")
    print(f"Input mode: {result['input_mode']}")
    print(f"Path: {result['path']}")
    print(f"Target format: {result['requested_target_format']}")
    if result.get("load_mode"):
        print(f"Checkpoint load mode: {result['load_mode']}")
    if result.get("load_warnings"):
        print("Load warnings:")
        _print_list(result["load_warnings"])

    print_section("Detected source")
    print(f"Framework: {detected['framework']}")
    print(f"Model family: {detected['model_family']}")
    print(f"Task: {detected['task']}")
    print(f"Status: {detected['status']}")
    print(f"Confidence: {detected['confidence']}")
    print("Evidence:")
    _print_list(detected.get("evidence", []))

    print_section("Official source exporter")
    print(f"Status: {official['status']}")
    print(f"Provider: {official['provider'] or '-'}")
    print(f"Target format: {official['target_format']}")
    print(f"Confidence: {official['confidence']}")
    print("Evidence:")
    _print_list(official.get("evidence", []))
    print("Blockers:")
    _print_list(official.get("blockers", []))
    print("Unknowns:")
    _print_list(official.get("unknowns", []))

    print_section("Toolkit generic exporter")
    print(f"Status: {toolkit['status']}")
    print(f"Target format: {toolkit['target_format']}")
    print(f"Confidence: {toolkit['confidence']}")
    print("Evidence:")
    _print_list(toolkit.get("evidence", []))
    print("Blockers:")
    _print_list(toolkit.get("blockers", []))
    print("Unknowns:")
    _print_list(toolkit.get("unknowns", []))

    print_section("Recommended route")
    print(f"Route: {recommendation['route']}")
    print(f"Confidence: {recommendation['confidence']}")
    print("Reason:")
    _print_list(recommendation.get("reason", []))

    print_section("Next suggested action")
    _print_list(_next_suggested_actions(result))


def _print_list(values: list[Any]) -> None:
    if values:
        for value in values:
            print(f"- {value}")
    else:
        print("- None")


def _next_suggested_actions(result: dict[str, Any]) -> list[str]:
    route = result["recommended_route"]["route"]
    provider = result["official_source_exporter"].get("provider")
    target_format = result["requested_target_format"]
    if route == "official_source_exporter":
        provider_label = provider or "source framework"
        return [
            f"use the {provider_label} official {target_format} exporter first",
            "then validate the produced ONNX with validate-onnx",
        ]
    if route == "toolkit_generic_exporter":
        return ["use edge-inspect export-onnx with the assessed spec"]
    if result["input_mode"] == "checkpoint":
        return ["create or refine a spec before attempting export"]
    return ["review blockers and unknowns before choosing an export route"]


def load_checkpoint(path: Path, unsafe_load: bool = False) -> Any:
    import torch

    supports_weights_only = _torch_load_supports_weights_only(torch.load)

    if unsafe_load:
        print(
            "Warning: unsafe loading uses Python pickle and should only be used "
            "for trusted local files.",
            file=sys.stderr,
        )
        if supports_weights_only:
            return torch.load(path, map_location="cpu", weights_only=False)
        return torch.load(path, map_location="cpu")

    if supports_weights_only:
        return torch.load(path, map_location="cpu", weights_only=True)

    print(
        "Warning: installed PyTorch does not support weights_only=True; falling "
        "back to torch.load(..., map_location='cpu'). Only inspect trusted files.",
        file=sys.stderr,
    )
    return torch.load(path, map_location="cpu")


def _torch_load_supports_weights_only(torch_load: Any) -> bool:
    try:
        return "weights_only" in inspect.signature(torch_load).parameters
    except (TypeError, ValueError):
        return False


if __name__ == "__main__":
    raise SystemExit(main())
