"""Un JSON atomico per checkpoint; PNG e impronte per le immagini."""

import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def environment():
    packages = {}
    for name in ("numpy", "Pillow", "torch", "diffusers", "transformers", "accelerate", "dlib", "scipy"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    source = {p.name: sha256(p) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return {"python": platform.python_version(), "platform": platform.platform(),
            "packages": packages, "source_sha256": source}


def verify_images(root, state):
    for record in state["records"]:
        for item in record["images"]:
            if sha256(Path(root) / item["path"]) != item["sha256"]:
                raise ValueError(f"Immagine modificata: {item['path']}")
