"""Configurazione esplicita: nessun parametro nascosto nel ciclo."""

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path


@dataclass
class Config:
    target: str = "flux"
    mode: str = "test"
    images: int = 2
    iterations: int = 5  # B: tentativi di ricerca, senza arresto anticipato
    verification_batches: int = 5  # V: batch nuovi per ciascun prompt congelato
    protocol: str = "search-verification-v1"
    memory: int = 2
    seed: int = 42
    max_generations: int = 300
    attacker: str = "ministral-3:8b"
    judge: str = "qwen3-vl:8b-instruct-q4_K_M"
    ollama_url: str = "http://localhost:11434"
    attacker_temperature: float = 1.0
    device: str = "auto"
    dtype: str = "bfloat16"
    width: int = 1024
    height: int = 1024
    steps: int = 16
    revision: str = "main"
    sequential: bool = True
    cpu_offload: bool = True
    available_memory_gb: float = 0
    target_estimate_gb: float = 0
    attacker_estimate_gb: float = 7
    judge_estimate_gb: float = 8
    mock: bool = False

    def validate(self):
        if self.target not in {"flux", "qwen"}:
            raise ValueError("target deve essere flux o qwen")
        if self.protocol != "search-verification-v1":
            raise ValueError("Protocollo incompatibile con questo branch")
        if min(self.images, self.iterations, self.verification_batches, self.memory, self.max_generations, self.steps) < 1:
            raise ValueError("Iterazioni, memoria, budget e passi devono essere positivi")
        if self.width < 64 or self.height < 64 or self.width % 16 or self.height % 16:
            raise ValueError("Risoluzione: multipli di 16, almeno 64")
        if max(self.iterations, self.verification_batches) * self.images >= 2**31:
            raise ValueError("Troppi campioni per lo spazio dei semi")
        if self.device not in {"auto", "cuda", "mps", "cpu"}:
            raise ValueError("Device sconosciuto")
        if self.dtype not in {"bfloat16", "float16", "float32"}:
            raise ValueError("dtype sconosciuto")
        if self.available_memory_gb:
            if self.target_estimate_gb <= 0:
                raise ValueError("Specificare target_estimate_gb per il controllo statico della memoria")
            footprints = [self.target_estimate_gb, self.attacker_estimate_gb, self.judge_estimate_gb]
            peak = max(footprints) if self.sequential else sum(footprints)
            if peak > self.available_memory_gb:
                raise ValueError(f"Picco stimato {peak:g} GiB > budget {self.available_memory_gb:g} GiB")

    @property
    def model_id(self):
        return {"flux": "black-forest-labs/flux-3-image-klein", "qwen": "Qwen/Qwen-Image-2.1"}[self.target]


def load_config(path):
    values = json.loads(Path(path).read_text())
    if "quorum" in values or "baseline" in values:
        raise ValueError("Configurazione precedente: creare un nuovo file con init-config")
    config = Config(**values)
    config.validate()
    return config


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def experiment_id(config, professions, provenance):
    return fingerprint({"config": asdict(config), "professions": professions, "provenance": provenance})[:16]
