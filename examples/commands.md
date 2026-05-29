# Command Examples

These examples use the product CLI entrypoint:

```bash
edge-inspect --help
edge-inspect inspect model.pt
edge-inspect inspect model.ckpt --unsafe-load --format markdown --output reports/model.md
```

Inspect and validate ONNX:

```bash
edge-inspect validate-onnx model.onnx --input-shape 1,3,640,640
edge-inspect validate-onnx model.onnx --input-shape 1,3,640,640 --target tensorrt-orin-nano --format markdown --output reports/model_readiness.md
edge-inspect validate-onnx model.onnx --input-shape 1,3,640,640 --target onnxruntime-cpu --json reports/model.json
```

Batch scan and compare:

```bash
edge-inspect scan ./models --include "*.onnx" --target tensorrt-orin-nano --input-shape 1,3,640,640 --format markdown --output reports/scan.md
edge-inspect compare old.onnx new.onnx --format json --output reports/compare.json
```

CI-style gating:

```bash
edge-inspect validate-onnx model.onnx --input-shape 1,3,640,640 --target tensorrt-orin-nano --fail-on warning
edge-inspect scan ./models --include "*.onnx" --target tensorrt-orin-nano --fail-on error
```

Backward-compatible module invocation remains available:

```bash
python -m converter.cli validate-onnx model.onnx --input-shape 1,3,640,640
```

This project does not execute source-framework exporters, run `trtexec`, or
build TensorRT engines from these commands.
