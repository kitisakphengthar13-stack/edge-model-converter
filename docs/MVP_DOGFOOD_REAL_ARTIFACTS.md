# MVP Dogfood: Real Artifacts

Dogfood run:

```text
artifacts/mvp_dogfood_20260529-154051
```

This run used only the product-facing `edge-inspect` CLI for inspection,
validation, scan, and compare commands. The only `python -m converter.cli`
command was the backward-compatibility help check.

## Commands Exercised

Environment and help:

```bash
edge-inspect --help
edge-inspect inspect --help
edge-inspect validate-onnx --help
edge-inspect scan --help
edge-inspect compare --help
python -m converter.cli --help
python -m pytest --basetemp artifacts/pytest_tmp
```

Checkpoint inspection:

```bash
edge-inspect inspect <model>
edge-inspect inspect <model> --unsafe-load --format markdown --output <report.md> --json <report.json>
```

ONNX readiness validation:

```bash
edge-inspect validate-onnx <model.onnx> --input-shape <shape> --target tensorrt-orin-nano --format markdown --output <report.md> --json <report.json>
```

Scan and compare:

```bash
edge-inspect scan <onnx-glob-or-dir> --input-shape <shape> --target tensorrt-orin-nano --format markdown --output <scan.md>
edge-inspect compare old.onnx new.onnx --format text --output compare.txt
edge-inspect compare old.onnx new.onnx --format json --output compare.json
```

## Checkpoint Outcomes

| Artifact | Safe Inspect | Unsafe Inspect | Framework | Top Task | Notes |
|---|---:|---:|---|---|---|
| `patchcore_anomalib/model.ckpt` | failed closed | passed | `anomalib` | `anomaly_detection` | PyTorch Lightning checkpoint; tensor summary available. |
| `patchcore_anomalib/model.pt` | failed closed | passed | `anomalib` | `-` | Generic dict; little inspectable tensor content. |
| `rd_anomalib/model.pt` | failed closed | passed | `anomalib` | `-` | Generic dict; little inspectable tensor content. |
| `yolo_ultralytics/best.pt` | failed closed | passed | `ultralytics` | `object_detection` | Framework and task hints detected from checkpoint metadata. |
| `yolo_ultralytics/yolo26s-seg.pt` | failed closed | passed | `ultralytics` | `segmentation` | Segmentation now outranks detection when mask/seg evidence is present. |

All safe inspection failures were expected because these real framework files
require trusted-local pickle loading under current PyTorch `weights_only`
security behavior.

## ONNX Readiness Outcomes

Target: `tensorrt-orin-nano`.

| Artifact | Checker | ORT Session | Dummy Inference | Nodes | Params | Readiness | Score | Main Findings |
|---|---:|---:|---:|---:|---:|---|---:|---|
| YOLO detection `best.onnx` | passed | passed | passed | 453 | 9,467,115 | caution | 90 | FP32-only initializers. |
| YOLO segmentation `yolo26s-seg.onnx` | passed | passed | passed | 535 | 10,396,300 | caution | 80 | FP32-only initializers; segmentation-like output tensors. |
| Anomalib PatchCore `patchcore_ckpt.onnx` | passed | passed | passed | 353 | 864,753,221 | blocked | 0 | External data, >2 GB initializer memory, >500M params, dynamic shapes, FP32-only, large memory bank. |

## Reports To Review

Generated reports live under:

```text
artifacts/mvp_dogfood_20260529-154051/reports
artifacts/mvp_dogfood_20260529-154051/scans
artifacts/mvp_dogfood_20260529-154051/compares
artifacts/mvp_dogfood_20260529-154051/logs
```

Key files:

- `reports/*_inspect.md` and `reports/*_inspect.json`
- `reports/*_readiness.md` and `reports/*_readiness.json`
- `scans/yolo_scan.md`
- `scans/patchcore_scan.md`
- `compares/yolo_det_vs_seg.txt`
- `compares/yolo_det_vs_seg.json`

## UX Findings

- Safe-load failures are actionable, but raw PyTorch errors are verbose.
- Unsafe-load warnings are clear and explicit about trusted-local pickle loading.
- `inspection_status` and readiness are separated correctly; PatchCore validates
  successfully while readiness is `blocked`.
- Scan Markdown is readable enough for short runs, but long Windows paths make
  tables wide.
- JSON report shapes are consistent enough for CI consumers.
- `--fail-on warning|error` remains predictable because readiness findings are
  promoted into structured report messages.
- Scan currently accepts one shared `--input-shape`, so mixed-shape ONNX sets
  should be scanned by shape group.

## Polish Applied

- Reduced duplicate top-level ONNX messages when readiness findings already
  supersede legacy TensorRT risk hints.
- Adjusted checkpoint task ranking so segmentation outranks object detection
  when YOLO-style mask/segmentation evidence and box evidence are both present.
- Added tests for both behavior changes.

## Known Limitations

- `scan` has one shared input shape. Use separate scans for mixed-shape model
  sets.
- Unsafe PyTorch loading is required for many real framework checkpoints.
- Source-framework export remains external by design.
- Readiness profiles are advisory. They do not build engines, run `trtexec`, or
  prove latency/accuracy on target hardware.
- Compare currently supports ONNX-vs-ONNX first.
- Large real model artifacts should remain local and should not be committed.

## MVP Acceptance

The MVP is accepted for local dogfooding:

- `edge-inspect` works for inspect, validate, scan, compare, and help flows.
- `python -m converter.cli --help` remains backward compatible.
- Synthetic tests pass.
- Real checkpoint safe/unsafe behavior is correct.
- Real YOLO and PatchCore ONNX readiness results match expected outcomes.
- No source model files were modified.
- No source-framework exporter execution was added.
- No `trtexec` command was run and no TensorRT engines were built.
