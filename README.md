# Edge Model Inspection and ONNX Preparation Toolkit

A local CLI toolkit for inspecting PyTorch checkpoints and ONNX artifacts before
edge deployment. It helps engineers safely understand opaque `.pt`, `.pth`,
`.ckpt`, and `.onnx` files by reporting checkpoint structure, metadata, tensor
shapes, dtypes, parameter counts, large tensors, framework hints, export-route
recommendations, and validation results.

This project does **not** claim to automatically convert every `.pt`, `.pth`, or
`.ckpt` file into ONNX. PyTorch checkpoint files can store very different
things, and some models are best exported through their original framework.

## What This Solves

PyTorch model files are often ambiguous: a file may be a raw `state_dict`, a
training checkpoint, a full model object, a PyTorch Lightning checkpoint, or a
framework-specific artifact. This toolkit makes that uncertainty explicit and
turns it into a repeatable ONNX preparation workflow:

- inspect checkpoint structure and deployment signals
- summarize top-level keys, nested model/checkpoint sections, tensor counts,
  parameter counts, dtype histograms, and large tensors
- inspect ONNX graph metadata, inputs, outputs, initializers, operators,
  dynamic dimensions, external data, and target-aware edge readiness risks
- validate a model spec before running trusted code
- assess whether an official source-library ONNX exporter should be used
- attempt generic PyTorch-to-ONNX export only when the spec is sufficient
- validate ONNX files with ONNX checker and ONNX Runtime inference
- score ONNX artifacts against advisory readiness profiles for generic,
  TensorRT Orin Nano, ONNX Runtime CPU, and OpenVINO deployment paths

## Quickstart

```bash
python -m pip install -e ".[dev]"

edge-inspect inspect model.pt
edge-inspect validate-onnx model.onnx --target tensorrt-orin-nano --input-shape 1,3,640,640
edge-inspect scan ./models --include "*.onnx" --target tensorrt-orin-nano --input-shape 1,3,640,640
edge-inspect compare old.onnx new.onnx
```

Use `python -m converter.cli ...` when invoking the module directly is more
convenient or when preserving older scripts.

## What This Project Is / Is Not

This project is:

- a PyTorch checkpoint inspection toolkit
- a local edge deployment-readiness aid
- a spec-driven ONNX preparation workflow
- an export-route assessment tool
- a guarded generic PyTorch-to-ONNX fallback exporter
- an ONNX validation tool

This project is not:

- a universal `.pt` / `.pth` / `.ckpt` converter
- a replacement for Anomalib, Ultralytics, or other source frameworks
- an automatic exporter for every model architecture
- a TensorRT engine builder
- a collection of runtime-specific deployment backends

## Core Workflow

1. Inspect a PyTorch checkpoint.
2. Describe model construction, checkpoint loading, input, output, and export
   intent in a spec.
3. Assess the recommended ONNX export route.
4. Use the official source-library ONNX exporter first when available.
5. Use this toolkit's generic PyTorch-to-ONNX exporter only when the spec has
   enough information and a dry-run forward pass succeeds.
6. Validate the resulting ONNX artifact with ONNX checker and ONNX Runtime.

## Export Strategy

The project uses a source-first ONNX export strategy.

Prefer the official exporter from the source framework/library:

- Anomalib models should use Anomalib export when available.
- Ultralytics YOLO models should use Ultralytics export when available.
- Other framework-specific models should use their official ONNX path when one
  exists.

Use this toolkit's generic PyTorch-to-ONNX exporter only as a fallback when:

- `model.module` and `model.class_name` are available
- checkpoint loading is defined
- input and output specs are provided
- a guarded PyTorch dry-run forward pass succeeds

Official source-library exporters and the toolkit generic exporter are separate
routes. The goal is not to force every model through one converter; it is to
produce a reliable, validated ONNX artifact through the most appropriate path.

## Export Capability Assessment

`assess-export` is a non-executing analysis command. It does not import model
code, instantiate models, load checkpoints from specs, run external exporters,
or create artifacts.

It reports:

- detected or declared source framework and model family
- whether an official source-library ONNX exporter route is known or likely
- whether this toolkit's generic PyTorch-to-ONNX exporter can be attempted
- recommended route, evidence, blockers, and unknowns

Examples:

```bash
edge-inspect assess-export specs/patchcore_cable_coreset_0_1.yaml
edge-inspect assess-export specs/yolo26n_task_detect.yaml
edge-inspect assess-export specs/example_simple_classifier_dryrun.yaml
```

Checkpoint-path assessment is preliminary and lower confidence because a
checkpoint alone usually does not contain the full export contract:

```bash
edge-inspect assess-export path/to/model.pt --unsafe-load
```

## Main CLI Examples

Install in editable mode during development:

```bash
python -m pip install -e ".[dev]"
```

Use `edge-inspect` for product-facing commands. The older
`python -m converter.cli ...` entrypoint remains backward compatible.

```bash
edge-inspect inspect models/model.pt
edge-inspect inspect models/model.ckpt --max-items 120
edge-inspect inspect models/model.pt --unsafe-load
edge-inspect inspect models/model.pt --unsafe-load --json artifacts/model_inspection.json
edge-inspect inspect models/model.pt --unsafe-load --format markdown --output artifacts/model_inspection.md

edge-inspect validate-spec specs/example_simple_classifier_dryrun.yaml
edge-inspect assess-export specs/example_simple_classifier_dryrun.yaml
edge-inspect plan-load specs/patchcore_cable_coreset_0_1.yaml

edge-inspect dry-run-model specs/example_simple_classifier_dryrun.yaml --allow-imports
edge-inspect export-onnx specs/example_simple_classifier_dryrun.yaml --allow-imports
edge-inspect validate-onnx artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml
edge-inspect validate-onnx artifacts/simple_classifier_dryrun/model.onnx --input-shape 1,3,2,2 --json artifacts/simple_classifier_onnx_report.json
edge-inspect validate-onnx artifacts/simple_classifier_dryrun/model.onnx --input-shape 1,3,2,2 --format markdown --output artifacts/simple_classifier_onnx_report.md
edge-inspect validate-onnx artifacts/simple_classifier_dryrun/model.onnx --input-shape 1,3,2,2 --target tensorrt-orin-nano --format markdown --output artifacts/simple_classifier_readiness.md

edge-inspect scan artifacts/simple_classifier_dryrun --include "*.onnx" --input-shape 1,3,2,2 --target tensorrt-orin-nano --format markdown --output artifacts/scan.md
edge-inspect compare artifacts/simple_classifier_dryrun/model.onnx artifacts/simple_classifier_dryrun/simple_classifier.onnx --format json --output artifacts/compare.json
```

`dry-run-model` and `export-onnx` may import local user code, so both are
guarded with explicit flags. Use them only for trusted local modules.

## ONNX Validation

`validate-onnx` checks exported ONNX files before downstream deployment. It:

- loads the ONNX file
- reports file size, IR version, producer metadata, opset imports, graph counts,
  initializer memory, parameter count, dtype histogram, operator histogram, and
  custom/non-standard op domains
- reports input/output names, shapes, dtypes, and dynamic dimensions
- runs `onnx.checker.check_model`
- creates an ONNX Runtime CPU session
- builds dummy input from the spec or CLI arguments
- runs inference
- prints readable input/output summaries
- can write a structured JSON report with `--json`

It does not import user modules, load checkpoints, execute source-framework
exporters, or build deployment engines.

## Edge Readiness Profiles

ONNX validation and batch scan can attach an advisory readiness profile with
`--target`:

- `generic`
- `tensorrt-orin-nano`
- `onnxruntime-cpu`
- `openvino`

Readiness output includes a target, score, level (`pass`, `caution`, or
`blocked`), findings with severity/code/message, and recommended next actions.
The rules are intentionally conservative and do not replace real target-device
benchmarking or engine compilation.

The current rules flag:

- external ONNX data files on deployment targets
- initializer memory above 512 MB, and above 2 GB for TensorRT Orin Nano
- parameter count above 100M and 500M
- dynamic dimensions that require explicit TensorRT profiles
- custom/non-standard ONNX operator domains
- FP32-only initializers on edge targets
- skipped/failed ONNX checker, ONNX Runtime session, or dummy inference
- segmentation-like multi-output tensors
- very large PatchCore/anomaly memory-bank artifacts

Examples:

```bash
edge-inspect validate-onnx model.onnx --input-shape 1,3,640,640 --target tensorrt-orin-nano --format markdown --output reports/model_readiness.md
edge-inspect validate-onnx model.onnx --input-shape 1,3,640,640 --target onnxruntime-cpu --fail-on warning
edge-inspect scan artifacts/exports --include "*.onnx" --input-shape 1,3,640,640 --target tensorrt-orin-nano --format json --output reports/scan_readiness.json
```

## Reports, Batch Scan, and Compare

Single-artifact PyTorch inspection and ONNX validation can emit text, JSON, or
Markdown reports with `--format text|json|markdown` and `--output`. Existing
`--json` options remain available for backward-compatible structured reports.

`scan` batch-inspects supported checkpoint and ONNX artifacts under a file,
directory, or glob. It continues after per-file failures and can write aggregate
JSON or Markdown summaries:

```bash
edge-inspect scan models --include "*.pt" --include "*.onnx" --format markdown --output artifacts/model_scan.md
```

`compare` currently supports ONNX-vs-ONNX comparison. It reports differences in
file size, input/output signatures, node count, parameter count, dtype
histograms, operator histograms, dynamic-shape status, and warning counts.

`validate-onnx` and `scan` support `--fail-on warning|error` for CI-style gating.

## Real Case Studies

### PatchCore / Anomalib

See [docs/PATCHCORE_REAL_TEST.md](docs/PATCHCORE_REAL_TEST.md) and
`specs/patchcore_cable_coreset_0_1.yaml`.

A real Anomalib PatchCore checkpoint was inspected, exported to ONNX with the
official Anomalib exporter, and validated with this toolkit. `assess-export`
recommends the official Anomalib ONNX route first, while the toolkit generic
exporter is intentionally blocked for the current spec because module/class
construction is not provided.

### YOLO / Ultralytics

See [docs/YOLO_REAL_TEST.md](docs/YOLO_REAL_TEST.md) and
`specs/yolo26n_task_detect.yaml`.

A real Ultralytics YOLO26n detection checkpoint was inspected, exported to ONNX
with the official Ultralytics exporter, and validated with this toolkit.
`assess-export` recommends the official Ultralytics ONNX route first, while the
toolkit generic exporter is not the appropriate path for that spec.

## Optional Downstream TensorRT Planning

`plan-tensorrt` is a small downstream helper that generates a `trtexec` planning
command from an existing ONNX artifact. It does not run `trtexec`, does not
require TensorRT, and does not build engine files.

```bash
edge-inspect plan-tensorrt artifacts/simple_classifier_dryrun/model.onnx --spec specs/example_simple_classifier_dryrun.yaml --target orin_nano --precision fp16
```

TensorRT engine creation is outside the core ONNX workflow and should happen on
the target device or a matching runtime environment.

## Current Limitations

- Unsafe PyTorch checkpoint loading uses Python pickle behavior and is trusted
  local only. Safe `weights_only=True` loading is attempted by default when the
  installed PyTorch version supports it.
- Checkpoint-only files often do not contain enough information for generic
  ONNX export. A model construction spec, checkpoint loading rule, input spec,
  and output spec are required before the guarded generic exporter can run.
- Source-framework exporters such as Ultralytics and Anomalib are recommended
  when detected, but they are not executed by this toolkit.
- JSON and Markdown output are available for PyTorch checkpoint inspection,
  ONNX validation/inspection, and batch scan. HTML reports are not implemented
  yet.
- ONNX validation requires an existing ONNX artifact. The tool does not claim an
  export occurred unless a file is actually produced.
- TensorRT support is planning only; `trtexec` is not executed and engines are
  not built.
- Readiness profiles are advisory only. They do not build TensorRT engines, run
  `trtexec`, perform OpenVINO conversion, or prove latency/accuracy on target
  hardware.

## Project Boundaries

- ONNX is the core artifact format for this project.
- Source-framework exporters are preferred when they are the authoritative route
  for a model family.
- Generic export is intentionally guarded and spec-driven.
- Real exported ONNX artifacts may be kept local-only when they are large or
  machine-specific.
