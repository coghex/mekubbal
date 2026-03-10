from __future__ import annotations

import hashlib
import json
import random
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file_sha256(path: str | Path) -> str:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(f"Cannot hash missing file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def object_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dependency_versions() -> dict[str, str]:
    package_names = ["numpy", "pandas", "torch", "stable-baselines3", "gymnasium"]
    versions: dict[str, str] = {}
    for name in package_names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "not-installed"
    return versions


def write_manifest(manifest_path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(manifest_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def python_version() -> str:
    return sys.version.split()[0]

