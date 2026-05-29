# Changelog

## 0.1.0 - MVP Release

- Completed Milestones 1-7 for the first dogfooded MVP.
- Verified `edge-inspect` on real PyTorch checkpoints and real ONNX artifacts.
- Confirmed checkpoint inspection, ONNX readiness validation, scan, compare,
  report, and CI-gating workflows.
- Added `edge-inspect --version`.

## Milestone 6: Product Hardening and CLI UX Polish

- Added the `edge-inspect` console entrypoint.
- Added packaging metadata in `pyproject.toml`.
- Normalized README examples around the product CLI.
- Added example command documentation and sample report shape notes.
- Added CI and release checklist documentation.

## Milestone 5: Edge Readiness Profiles

- Added advisory readiness profiles for `generic`, `tensorrt-orin-nano`,
  `onnxruntime-cpu`, and `openvino`.
- Added target-aware readiness findings, scores, levels, and next actions.
- Integrated readiness into ONNX validation, reports, scan summaries, and CI
  severity gating.

## Milestone 4: Source-Framework ONNX Artifact Reality Test

- Validated real ONNX artifacts exported by official Ultralytics and Anomalib
  paths outside the project CLI.
- Confirmed YOLO detection and segmentation ONNX validation and TensorRT
  planning behavior.
- Identified PatchCore external-data and memory-bank readiness blockers.

## Milestone 3: Reports, Batch Scan, and Compare MVP

- Added report envelopes, Markdown output, batch scan, ONNX compare, and
  CI-friendly `--fail-on` severity behavior.

## Milestone 2: ONNX Artifact Inspector MVP

- Added structured ONNX inspection with graph metadata, inputs, outputs,
  initializer summaries, operator histograms, checker status, ONNX Runtime
  session/inference status, dynamic shape warnings, external data detection, and
  TensorRT risk hints.

## Milestone 1: Real Checkpoint Inspector MVP

- Added structured PyTorch checkpoint reports, JSON output, top-level and nested
  key summaries, tensor summaries, parameter counts, dtype histograms, framework
  detection, improved export-route assessment, and tests.
