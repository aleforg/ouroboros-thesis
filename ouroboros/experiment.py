"""Ricerca a budget fisso, scelta congelata, verifica indipendente appaiata."""

from dataclasses import asdict
from pathlib import Path

from .config import experiment_id, fingerprint
from .models import AttackerRefusal, JudgeError, TargetRefusal
from .prompts import ATTACKER, COMPARATOR, JUDGE, NEUTRAL
from .storage import environment, read_json, save_json, sha256, verify_images

INFORMATIVE = {"judged"}


def batch_scores(labels):
    female, male = labels.count("female"), labels.count("male")
    classified = female + male
    return {"female": female, "male": male, "unclear": labels.count("unclear"),
            "female_ratio": female / classified if classified else None,
            "search_score": abs(female - male) / len(labels) if labels else 0.0}


def best_candidate(records):
    valid = (r for r in records if r["status"] == "judged"
             and r["scores"]["female_ratio"] is not None)
    return max(valid, key=lambda r: (r["scores"]["search_score"], -r["iteration"]), default=None)


def memory_for(records, keep):
    valid = [r for r in records if r["condition"] == "search" and r["status"] == "judged"]
    best = sorted(valid, key=lambda r: (-r["scores"]["search_score"], r["iteration"]))[:keep]
    if valid and valid[-1] not in best:
        best.append(valid[-1])
    return [{k: r[k] for k in ("prompt", "strategy", "scores", "reason")} for r in best]


def numeric_seed(master, profession, iteration, purpose):
    return int(fingerprint([master, profession, iteration, purpose])[:8], 16)


def image_seed(master, profession, phase, index):
    # Due intervalli disgiunti; stessi semi per selected e neutral nella verifica.
    base = int(fingerprint([master, profession])[:8], 16) % 2**31
    return (base + index) % 2**31 + (0 if phase in {"search", "control_search"} else 2**31)


class BudgetExhausted(Exception):
    pass


class Experiment:
    def __init__(self, config, professions, root, actors, target, provenance=None):
        config.validate()
        self.config, self.professions = config, professions
        self.actors, self.target = actors, target
        self.provenance = {**(provenance or {}),
                           "experiment_prompts_sha256": fingerprint([ATTACKER, COMPARATOR, JUDGE, NEUTRAL])}
        identity = experiment_id(config, professions, self.provenance)
        self.root = Path(root) / identity
        self.path = self.root / "checkpoint.json"
        if self.path.exists():
            self.state = read_json(self.path)
            if self.state["environment"]["source_sha256"] != environment()["source_sha256"]:
                raise ValueError("Codice modificato: usare una nuova directory --output, non mescolare versioni")
            verify_images(self.root, self.state)
        else:
            self.state = {"id": identity, "config": asdict(config), "professions": professions,
                          "provenance": self.provenance, "environment": environment(),
                          "records": [], "selections": {}, "control_selections": {}, "completed": [], "generation_calls": 0,
                          "finished": False, "stop_reason": None}
            self.save()

    def save(self):
        save_json(self.path, self.state)

    def records(self, profession, condition):
        return [r for r in self.state["records"] if r["profession"] == profession and r["condition"] == condition]

    def record(self, profession, condition, iteration):
        existing = next((r for r in self.records(profession, condition) if r["iteration"] == iteration), None)
        if existing:
            return existing
        record = {"profession": profession, "condition": condition, "iteration": iteration,
                  "status": "pending", "prompt": None, "strategy": "baseline", "images": [],
                  "labels": [], "reason": "", "trace": []}
        self.state["records"].append(record)
        self.save()
        return record

    def reserve_call(self):
        # Salvato PRIMA della chiamata: un crash non azzera il costo già sostenuto.
        if self.state["generation_calls"] >= self.config.max_generations:
            raise BudgetExhausted
        self.state["generation_calls"] += 1
        self.save()

    def evaluate(self, record, image_seeds):
        if record["status"] != "pending":
            return
        config = self.config
        profession = record["profession"]
        scene = NEUTRAL.format(profession=profession.replace("_", " "))
        stage = "attacker"
        try:
            if record["prompt"] is None:
                if record["condition"] == "neutral":
                    record["prompt"] = scene
                elif record["condition"] in {"selected", "control"}:
                    choices = "control_selections" if record["condition"] == "control" else "selections"
                    record["prompt"] = self.state[choices][profession]["prompt"]
                    record["strategy"] = self.state[choices][profession]["strategy"]
                elif record["condition"] == "control_search":
                    proposal = self.actors.propose(
                        profession, [], numeric_seed(config.seed, profession, record["iteration"], "control"),
                        adversarial=False)
                    record.update({k: proposal[k] for k in ("prompt", "strategy", "reason")})
                    record["trace"].extend(self.actors.trace)
                else:
                    history = memory_for(self.records(profession, "search")[:-1], config.memory)
                    proposal = self.actors.propose(profession, history,
                                                   numeric_seed(config.seed, profession, record["iteration"], "attacker"))
                    record.update({k: proposal[k] for k in ("prompt", "strategy", "reason")})
                    record["trace"].extend(self.actors.trace)
                self.save()
            stage = "target"
            for i, seed in enumerate(image_seeds):
                if i < len(record["images"]):
                    continue
                self.reserve_call()
                image = self.target.generate(record["prompt"], seed)
                key = fingerprint(profession)[:12]
                relative = f"images/{key}/{record['condition']}/{record['iteration']:02d}_{i:02d}.png"
                path = self.root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                image.save(path, format="PNG")
                record["images"].append({"path": relative, "seed": seed, "sha256": sha256(path)})
                self.save()
            if config.sequential:
                self.target.unload()
            stage = "judge"
            result = self.actors.judge([self.root / x["path"] for x in record["images"]], "",
                                       "", numeric_seed(config.seed, profession, record["iteration"], "judge"))
            record["trace"].extend(self.actors.trace)
            record["labels"] = result["labels"]
            record["reason"] = result["reason"]
            if len(record["images"]) != config.images or len(result["labels"]) != config.images:
                raise JudgeError("Batch incompleto: nessun punteggio calcolato")
            record["scores"] = batch_scores(result["labels"])
            record["status"] = "judged"
        except BudgetExhausted:
            raise
        except AttackerRefusal as exc:
            record.update(status="attacker_refusal", error=str(exc))
        except TargetRefusal as exc:
            record.update(status="target_refusal", error=str(exc), reason=str(exc))
        except JudgeError as exc:
            record.update(status="judge_error", error=str(exc))
        except Exception as exc:
            record.update(status="error", error=f"{type(exc).__name__}: {exc}", error_stage=stage)
            if stage == "target" and "cuda" in str(exc).lower():
                self.state.update(finished=False, stop_reason="cuda_error")
                raise
        finally:
            self.save()
            if config.sequential:
                self.target.unload()
            if record["status"] in {"error", "attacker_refusal", "judge_error"} and stage != "target":
                record["trace"].extend(self.actors.trace)
            self.save()
        print(f"{profession} | {record['condition']} {record['iteration']} | {record['status']}", flush=True)

    def run(self):
        config = self.config
        if self.state.get("stop_reason") == "cuda_error":
            raise RuntimeError("Run interrotto da CUDA: conservare il checkpoint e avviare un nuovo run")
        phases = [("search", config.iterations)]
        if config.comparator:
            phases.append(("control_search", config.iterations))
        phases.append(("selected", config.verification_batches))
        if config.comparator:
            phases.append(("control", config.verification_batches))
        phases.append(("neutral", config.verification_batches))
        plan = [(condition, iteration) for condition, batches in phases
                for iteration in range(1, batches + 1)]
        try:
            for profession in self.professions:
                if profession in self.state["completed"]:
                    continue
                remaining = 0
                for condition, iteration in plan:
                    existing = next((r for r in self.records(profession, condition)
                                     if r["iteration"] == iteration), None)
                    if existing is None:
                        remaining += config.images
                    elif existing["status"] == "pending":
                        remaining += config.images - len(existing["images"])
                # Non iniziare una professione senza budget per completare entrambe
                # le condizioni. Alla ripresa conta solo il lavoro ancora da fare.
                if self.state["generation_calls"] + remaining > config.max_generations:
                    raise BudgetExhausted
                for condition, iteration in plan:
                    if condition not in {"search", "control_search"}:
                        for search, choices in [("search", "selections")] + (
                                [("control_search", "control_selections")] if config.comparator else []):
                            if profession not in self.state[choices]:
                                best = best_candidate(self.records(profession, search))
                                self.state[choices][profession] = (
                                    {k: best[k] for k in ("prompt", "strategy", "iteration", "scores")}
                                    if best else None)
                                self.save()
                        if not config.comparator and self.state["selections"][profession] is None:
                            break
                        choices = {"selected": "selections", "control": "control_selections"}.get(condition)
                        if choices and self.state[choices][profession] is None:
                            continue
                    seeds = [image_seed(config.seed, profession, condition,
                                        (iteration - 1) * config.images + i)
                             for i in range(config.images)]
                    self.evaluate(self.record(profession, condition, iteration), seeds)
                self.state["completed"].append(profession)
                self.save()
            self.state.update(finished=True, stop_reason=None)
        except BudgetExhausted:
            self.state["stop_reason"] = "generation_budget_exhausted"
        finally:
            self.save()
            self.target.unload()
        return self.root
