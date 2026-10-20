"""Batch Modal: modal run modal_app.py --help. Nessuna risorsa parte all'import."""

import os
from pathlib import Path
import shlex

import modal

ROOT = Path(__file__).resolve().parent
GPU = os.environ.get("OUROBOROS_GPU", "L40S")
HOURS = float(os.environ.get("OUROBOROS_HOURS", "1"))
if not 0 < HOURS <= 23:
    raise ValueError("OUROBOROS_HOURS deve essere > 0 e <= 23")
SECONDS = int(HOURS * 3600)
if SECONDS < 60:
    raise ValueError("Durata minima: 60 secondi")

app = modal.App("ouroboros-thesis")
cache = modal.Volume.from_name("ouroboros-models", create_if_missing=True)
results = modal.Volume.from_name("ouroboros-results", create_if_missing=True)
# Secret opzionale: creare in Modal un Secret contenente HF_TOKEN se necessario.
secret_name = os.environ.get("OUROBOROS_HF_SECRET")
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []

base = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("numpy>=1.26", "Pillow>=10")
    .env({"PYTHONPATH": "/app", "PYTHONUNBUFFERED": "1"})
)
image = (
    base.apt_install("git", "curl", "zstd", "libgomp1")
    .pip_install("torch>=2.4")
    .pip_install_from_requirements(str(ROOT / "requirements-models.txt"))
    .run_commands(
        "curl -fsSL --retry 3 https://ollama.com/download/ollama-linux-amd64.tar.zst -o /tmp/ollama.tar.zst",
        "tar --zstd -xf /tmp/ollama.tar.zst -C /usr && rm /tmp/ollama.tar.zst",
    )
    .env({"HF_HOME": "/cache/huggingface", "OLLAMA_MODELS": "/cache/ollama",
          "OLLAMA_HOST": "127.0.0.1:11434", "OLLAMA_NUM_PARALLEL": "1"})
)


def sources(container):
    # Non caricare .venv, .git, credenziali, dataset di validazione o run locali.
    container = container.workdir("/app")
    for directory in ("ouroboros", "cloud", "configs"):
        container = container.add_local_dir(ROOT / directory, f"/app/{directory}")
    container = container.add_local_file(ROOT / "data/professions.txt", "/app/data/professions.txt")
    container = container.add_local_dir(ROOT / "data/subsets", "/app/data/subsets")
    return container


@app.function(image=sources(base), volumes={"/results": results}, timeout=300,
              max_containers=1, scaledown_window=2)
def smoke():
    import subprocess
    try:
        subprocess.run(["python", "-m", "ouroboros", "run", "--config", "configs/flux-test.json",
                        "--mock", "--limit", "1", "--output", "/results/mock"], check=True)
    finally:
        results.commit()


@app.function(image=sources(base), volumes={"/results": results}, timeout=600,
              max_containers=1, scaledown_window=2)
def prepare_validation():
    from ouroboros.validation import import_t2isafety
    results.reload()
    try:
        count = import_t2isafety(
            Path("/results/t2isafety/hf_test_fairness_generated.json"),
            Path("/results/t2isafety"), Path("/results/t2isafety/manifest.json"))
        print(f"Manifest remoto: {count} immagini")
    finally:
        results.commit()


@app.function(image=sources(image), gpu=GPU, cpu=2, memory=65536,
              volumes={"/cache": cache, "/results": results}, secrets=secrets,
              timeout=SECONDS + 120, max_containers=1, scaledown_window=2, retries=0)
def execute(arguments: list[str]):
    import json
    import subprocess
    import time
    from cloud.runtime import checked_process, wait_for_ollama
    from ouroboros.cli import parser
    from ouroboros.config import load_config

    args = parser().parse_args(arguments)
    config = load_config(args.config)
    if config.mock:
        raise ValueError("Per il mock usare --action smoke, senza GPU")
    if config.ollama_url.rstrip("/") not in {"http://localhost:11434", "http://127.0.0.1:11434"}:
        raise ValueError("Il container usa Ollama locale: correggere ollama_url nel preset")
    deadline = time.monotonic() + SECONDS
    results.reload()
    cache.reload()
    log_dir = Path("/results/environment") / str(time.time_ns())
    log_dir.mkdir(parents=True)
    for name, command in (("packages.txt", ["python", "-m", "pip", "freeze"]),
                          ("ollama.txt", ["ollama", "--version"]),
                          ("gpu.txt", ["nvidia-smi"])):
        with (log_dir / name).open("w") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, check=True)
    (log_dir / "invocation.json").write_text(json.dumps({"arguments": arguments, "gpu": GPU,
                                                       "max_seconds": SECONDS}, indent=2))
    results.commit()
    server = subprocess.Popen(["ollama", "serve"])
    try:
        wait_for_ollama(server)
        names = [config.judge] if args.command == "validate-judge" else [config.attacker, config.judge]
        for name in dict.fromkeys(names):
            # Riusa i pesi esistenti: non aggiornare i tag a ogni ripresa.
            found = subprocess.run(["ollama", "show", name], stdout=subprocess.DEVNULL,
                                   stderr=subprocess.DEVNULL).returncode == 0
            if not found:
                checked_process(["ollama", "pull", name], deadline=deadline, commit=cache.commit)
        checked_process(["python", "-m", "ouroboros", *arguments], deadline=deadline,
                        commit=results.commit)
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait()
        cache.commit()
        results.commit()


def main(action: str = "plan", config: str = "configs/flux-test.json",
         professions: str = "data/professions.txt", limit: int = 1,
         output: str = "/results/runs", validation: str = "",
         exploratory: bool = False, manifest: str = "", sample: int = 150):
    """action: plan, smoke, import-validation, doctor, run, validate-judge."""
    from ouroboros.config import load_config

    if action == "import-validation":
        prepare_validation.remote()
        return
    if action == "smoke":
        smoke.remote()
        return
    if action not in {"plan", "doctor", "run", "validate-judge"}:
        raise ValueError("Azione sconosciuta")
    # I percorsi config/professions sono relativi al repository caricato.
    for value, prefix in ((config, "configs"), (professions, "data")):
        path = Path(value)
        if not path.parts or path.is_absolute() or ".." in path.parts or path.parts[0] != prefix:
            raise ValueError(f"Percorso non ammesso: {value}")
    cfg = load_config(ROOT / config)
    available = [p for p in (ROOT / professions).read_text().splitlines() if p.strip()]
    if not 1 <= limit <= len(available):
        raise ValueError("limit deve essere positivo e non superare le professioni disponibili")
    command = "run" if action == "plan" else action
    arguments = [command, "--config", config]
    if command == "run":
        if not output.startswith("/results/") or ".." in Path(output).parts:
            raise ValueError("output deve essere contenuto in /results/")
        arguments += ["--professions", professions, "--limit", str(limit), "--output", output]
        if validation:
            if not validation.startswith("/results/") or ".." in Path(validation).parts:
                raise ValueError("validation deve essere contenuto in /results/")
            arguments += ["--validation", validation]
        if exploratory:
            arguments += ["--exploratory"]
        if not validation and not exploratory and action != "plan":
            raise ValueError("Specificare --validation oppure --exploratory per una prova preliminare")
    elif command == "validate-judge":
        if not manifest.startswith("/results/") or ".." in Path(manifest).parts:
            raise ValueError("manifest deve essere contenuto in /results/")
        if not output.startswith("/results/") or ".." in Path(output).parts:
            raise ValueError("output deve essere contenuto in /results/")
        arguments += ["--manifest", manifest, "--sample", str(sample), "--output", output]
    print(f"GPU: {GPU}; RAM: 64 GiB; durata massima: {HOURS:g} h (non è un tetto monetario)")
    print(f"Target: {cfg.model_id}; immagini pianificate: {limit * cfg.images * (cfg.iterations + 2 * cfg.verification_batches)}")
    print(shlex.join(["python", "-m", "ouroboros", *arguments]))
    if action != "plan":
        execute.remote(arguments)


entrypoint = app.local_entrypoint(name="main")(main)

if __name__ == "__main__":
    import argparse
    cli = argparse.ArgumentParser(description="Piano locale: nessuna chiamata a Modal")
    cli.add_argument("--config", default="configs/flux-test.json")
    cli.add_argument("--professions", default="data/professions.txt")
    cli.add_argument("--limit", type=int, default=1)
    main(action="plan", **vars(cli.parse_args()))
