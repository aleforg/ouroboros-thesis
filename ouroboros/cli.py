"""Interfaccia da terminale; il ciclo rimane leggibile in experiment.py."""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

from .config import Config, load_config
from .models import MockModels, Ollama, make_target, model_provenance
from .storage import environment, read_json, save_json, sha256


def parser():
    root = argparse.ArgumentParser(description="Ouroboros: esperimento sul bias di genere T2I")
    commands = root.add_subparsers(dest="command", required=True)
    init = commands.add_parser("init-config", help="Scrive una configurazione esplicita")
    init.add_argument("--target", choices=["flux", "qwen"], default="flux")
    init.add_argument("--mode", choices=["test", "full"], default="test")
    init.add_argument("--output", type=Path, required=True)
    run = commands.add_parser("run", help="Esegue o riprende un esperimento")
    run.add_argument("--config", type=Path, required=True)
    run.add_argument("--professions", type=Path, default=Path("data/professions.txt"))
    run.add_argument("--output", type=Path, default=Path("runs"))
    run.add_argument("--limit", type=int)
    run.add_argument("--mock", action="store_true", help="Solo test software, senza inferenza")
    run.add_argument("--validation", type=Path, help="Report validate-judge dello stesso giudice")
    run.add_argument("--exploratory", action="store_true", help="Run esplicitamente preliminare senza validazione")
    report = commands.add_parser("report", help="Calcola metriche, CSV e riepilogo Markdown")
    report.add_argument("run_dir", type=Path)
    report.add_argument("--cluster-strategies", action="store_true")
    importer = commands.add_parser("import-t2isafety", help="Converte le annotazioni ufficiali in un manifest")
    importer.add_argument("--annotations", type=Path, required=True)
    importer.add_argument("--image-root", type=Path, required=True)
    importer.add_argument("--output", type=Path, required=True)
    validation = commands.add_parser("validate-judge", help="Pilot su annotazioni umane")
    validation.add_argument("--config", type=Path, required=True)
    validation.add_argument("--manifest", type=Path, required=True)
    validation.add_argument("--sample", type=int, default=150)
    validation.add_argument("--output", type=Path, required=True)
    ff = commands.add_parser("fairface", help="Classifica tutte le immagini con FairFace")
    ff.add_argument("run_dir", type=Path)
    ff.add_argument("--weights", type=Path, required=True)
    ff.add_argument("--detector", type=Path, required=True)
    ff.add_argument("--landmarks", type=Path, required=True)
    ff.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    replay = commands.add_parser("replay", help="Confronta hash delle immagini rigenerate")
    replay.add_argument("run_dir", type=Path)
    replay.add_argument("--limit", type=int, default=8)
    doctor = commands.add_parser("doctor", help="Controlla dipendenze e modelli senza caricare pesi")
    doctor.add_argument("--config", type=Path, required=True)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        execute(args)
    except (ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"Errore: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


def execute(args):
    if args.command == "init-config":
        config = Config(target=args.target, mode=args.mode)
        if args.mode == "full":
            config.images, config.iterations, config.verification_batches, config.memory = 8, 5, 5, 3
            config.max_generations = 21000
        if args.target == "qwen":
            config.steps = 40
            config.width = config.height = 2048
        if args.output.exists():
            raise ValueError("Il file esiste già; modificarlo direttamente oppure scegliere un altro percorso")
        save_json(args.output, asdict(config))
        print(args.output)
    elif args.command == "run":
        from .experiment import Experiment
        from .validation import check_validation

        config = load_config(args.config)
        config.mock = config.mock or args.mock
        professions = [p.strip() for p in args.professions.read_text().splitlines() if p.strip()]
        if len(set(professions)) != len(professions):
            raise ValueError("Professioni duplicate")
        count = args.limit if args.limit is not None else (10 if config.mode == "test" else 175)
        if count < 1 or len(professions) < count:
            raise ValueError(f"Richieste {count} professioni, disponibili {len(professions)}")
        # Campione pseudo-casuale fisso per test; in full conserva l'intero elenco.
        import random
        professions = random.Random(config.seed).sample(professions, count)
        provenance = model_provenance(config)
        provenance["professions_sha256"] = sha256(args.professions)
        provenance["exploratory"] = args.exploratory
        if not config.mock and not args.exploratory:
            if args.validation is None:
                raise ValueError("Prima eseguire validate-judge e passare --validation; per prove usare --exploratory")
            provenance["validation_sha256"] = check_validation(config, args.validation, provenance)
        actors = MockModels() if config.mock else Ollama(config)
        target = make_target(config)
        experiment = Experiment(config, professions, args.output, actors, target, provenance)
        result = experiment.run()
        print(f"Run salvato: {result}")
        if not experiment.state["finished"]:
            print(f"Run incompleto: {experiment.state['stop_reason']}. Non trattarlo come un run completo.")
    elif args.command == "report":
        from .analysis import cluster_strategies, make_report

        result = make_report(args.run_dir)
        if args.cluster_strategies:
            cluster_strategies(args.run_dir)
        print(f"Report: {args.run_dir / 'report.md'}; coppie: {result['paired']['n_pairs']}")
    elif args.command == "import-t2isafety":
        from .validation import import_t2isafety

        print(f"Importate {import_t2isafety(args.annotations, args.image_root, args.output)} immagini")
    elif args.command == "validate-judge":
        from .validation import validate_judge

        config = load_config(args.config)
        provenance = model_provenance(config, judge_only=True)
        actors = MockModels() if config.mock else Ollama(config)
        try:
            report = validate_judge(config, actors, args.manifest, args.output, args.sample, provenance)
        finally:
            actors.unload()
        print(json.dumps(report["metrics"], indent=2))
    elif args.command == "fairface":
        from .fairface import annotate_run

        annotate_run(args.run_dir, args.weights, args.detector, args.landmarks, args.device)
    elif args.command == "replay":
        from .replay import replay

        config = Config(**read_json(args.run_dir / "checkpoint.json")["config"])
        config.validate()
        if config.backend == "sdcpp" and not config.mock:
            from .sdcpp import artifact_provenance
            if artifact_provenance(config) != read_json(args.run_dir / "checkpoint.json")["provenance"]["sdcpp"]:
                raise ValueError("Binario o pesi sd.cpp diversi dal run originale")
        target = make_target(config)
        result = replay(args.run_dir, target, args.limit)
        print(json.dumps(result["match_rate"], indent=2))
    elif args.command == "doctor":
        config = load_config(args.config)
        info = environment()
        info["models"] = model_provenance(config)
        if not config.mock and config.backend == "diffusers":
            import torch
            import diffusers

            name = "Flux2KleinPipeline" if config.target == "flux" else "QwenImage21Pipeline"
            if not hasattr(diffusers, name):
                raise RuntimeError(f"Diffusers non espone {name}")
            info["hardware"] = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available()}
        print(json.dumps(info, indent=2))
