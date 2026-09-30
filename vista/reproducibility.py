"""Seed initialization and local records for independently reproducible runs."""

import hashlib
import importlib.metadata
import json
import platform
import random
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import numpy as np
import torch
from omegaconf import OmegaConf


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_run_record(config):
    root = Path(__file__).resolve().parents[1]
    sources = sorted(
        p
        for name in ("vista", "verl", "agent_system", "scripts")
        for p in (root / name).rglob("*")
        if p.suffix in {".py", ".yaml", ".sh"}
    )
    digest = hashlib.sha256()
    for path in sources:
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    dependencies = {}
    for name in ("torch", "transformers", "vllm", "ray", "flash-attn", "tensordict"):
        try:
            dependencies[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            dependencies[name] = None
    now = datetime.now(timezone.utc)
    directory = (
        Path(config.trainer.default_local_dir)
        / "run_records"
        / (now.strftime("%Y%m%dT%H%M%SZ") + "-" + uuid4().hex[:8])
    )
    directory.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(config, directory / "config.yaml", resolve=True)
    (directory / "environment.json").write_text(
        json.dumps(
            {
                "created_utc": now.isoformat(),
                "seed": int(config.get("seed", 0)),
                "python": platform.python_version(),
                "dependencies": dependencies,
                "source_sha256": digest.hexdigest(),
            },
            indent=2,
        )
        + "\n"
    )
    return directory
