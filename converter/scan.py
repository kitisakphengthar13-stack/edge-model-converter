from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

from .cli_support import inspect_artifact
from .reports import ReportEnvelope, error_envelope, infer_artifact_kind


DEFAULT_INCLUDES = ("*.pt", "*.pth", "*.ckpt", "*.onnx")


def discover_files(path_or_glob: str, includes: list[str] | None = None) -> list[Path]:
    includes = includes or list(DEFAULT_INCLUDES)
    root = Path(path_or_glob)
    paths: list[Path] = []
    if any(ch in path_or_glob for ch in "*?[]"):
        paths.extend(Path(item) for item in glob.glob(path_or_glob, recursive=True))
    elif root.is_file():
        paths.append(root)
    elif root.is_dir():
        for pattern in includes:
            paths.extend(root.rglob(pattern))
    else:
        paths.extend(Path(item) for item in glob.glob(path_or_glob, recursive=True))
    return sorted({path for path in paths if path.is_file() and infer_artifact_kind(path) != "unsupported"})


def scan_artifacts(
    path_or_glob: str,
    *,
    includes: list[str] | None = None,
    unsafe_load: bool = False,
    input_shape: list[int] | None = None,
    max_items: int = 20,
    target: str = "generic",
) -> list[ReportEnvelope]:
    envelopes: list[ReportEnvelope] = []
    for path in discover_files(path_or_glob, includes=includes):
        try:
            envelopes.append(
                inspect_artifact(
                    path,
                    unsafe_load=unsafe_load,
                    input_shape=input_shape,
                    max_items=max_items,
                    target=target,
                )
            )
        except Exception as exc:
            envelopes.append(error_envelope(path, str(exc)))
    return envelopes
