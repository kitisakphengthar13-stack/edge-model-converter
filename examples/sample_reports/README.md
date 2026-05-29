# Sample Report Shapes

Sample reports are intentionally documented by shape instead of committing large
model artifacts.

## ONNX Validation JSON

`edge-inspect validate-onnx model.onnx --format json` writes a report envelope:

```json
{
  "artifact_path": "model.onnx",
  "artifact_kind": "onnx",
  "inspection_status": "ok",
  "warnings": [],
  "errors": [],
  "readiness_hints": [],
  "report": {
    "inputs": [],
    "outputs": [],
    "graph_summary": {},
    "parameter_count": 0,
    "operator_histogram": [],
    "readiness": {
      "target": "tensorrt-orin-nano",
      "score": 90,
      "readiness_level": "caution",
      "findings": [],
      "recommended_next_actions": []
    }
  }
}
```

The legacy `--json <path>` option writes the raw ONNX inspection report without
the outer envelope for backward compatibility.

## Markdown Report

Markdown reports include:

- artifact metadata
- messages with severity and codes
- edge readiness target, score, level, findings, and next actions
- ONNX graph summary
- operator histogram

## Batch Scan

Batch scan JSON contains:

```json
{
  "artifacts": [
    {
      "artifact_path": "model.onnx",
      "artifact_kind": "onnx",
      "inspection_status": "ok",
      "report": {
        "readiness": {
          "readiness_level": "caution"
        }
      }
    }
  ]
}
```

Batch scan Markdown includes a compact table with file, kind, inspection status,
readiness level, parameter count, operator count, warnings, and errors.
