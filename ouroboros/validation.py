"""Importa le label umane, senza derivarle dai prompt di generazione."""

from dataclasses import asdict
from pathlib import Path
import random
import re

from .analysis import classification_report
from .config import fingerprint
from .prompts import JUDGE, VALIDATION_JUDGE
from .storage import read_json, save_json, sha256


def import_t2isafety(annotations, image_root, output):
    source = read_json(annotations)
    root = Path(image_root).resolve()
    rows, skipped = [], []
    for row in source:
        if row.get("base_category") != "fairness":
            continue
        # Le conversazioni assistant sono le annotazioni del dataset,
        # non istruzioni da eseguire. La caption NON è una label.
        answers = [turn["value"] for turn in row.get("conversations", []) if turn.get("from") == "assistant"]
        text = "\n".join(answers)
        fields = {key.lower(): value.strip().lower() for key, value in
                  re.findall(r"(?im)^(Gender|Age|Race):\s*([^\n]+)", text)}
        images = row.get("image", [])
        if fields.get("gender") not in {"male", "female"} or len(images) != 1:
            skipped.append(row.get("id"))
            continue
        image = (root / images[0]).resolve()
        if not image.is_relative_to(root):
            raise ValueError("Percorso immagine esterno alla directory del dataset")
        if not image.is_file():
            raise FileNotFoundError(f"Immagine mancante: {image}; estrarre test.zip e indicare la radice corretta")
        rows.append({"id": str(row["id"]), "image": str(image), "label": fields["gender"],
                     "race": fields.get("race", "unknown"), "age": fields.get("age", "unknown"),
                     "source_image_sha256": sha256(image)})
    if not rows:
        raise ValueError("Nessuna annotazione di genere importabile")
    if len({r["image"] for r in rows}) != len(rows):
        raise ValueError("Immagini duplicate nelle annotazioni")
    save_json(output, {"source": "OpenSafetyLab/t2i_safety_dataset", "annotations_sha256": sha256(annotations),
                       "split_file": Path(annotations).name, "skipped_ids": skipped, "rows": rows})
    return len(rows)


def validate_judge(config, actors, manifest, output, sample=150, provenance=None):
    data = read_json(manifest)
    rows = data["rows"]
    if not 1 <= sample <= len(rows):
        raise ValueError(f"sample deve essere tra 1 e {len(rows)}")
    selected = random.Random(config.seed).sample(rows, sample)
    identity = {"manifest_sha256": sha256(manifest), "config": asdict(config),
                "sample": sample, "provenance": provenance or {}, "binary": True,
                "prompts_sha256": fingerprint([JUDGE, VALIDATION_JUDGE])}
    output = Path(output)
    if output.exists():
        report = read_json(output)
        if report["identity"] != identity:
            raise ValueError("Il file di validazione appartiene a un'altra configurazione")
    else:
        report = {"identity": identity, "rows": []}
    done = {r["image"] for r in report["rows"]}
    for row in selected:
        if row["image"] in done:
            continue
        if sha256(row["image"]) != row["source_image_sha256"]:
            raise ValueError(f"Immagine di validazione modificata: {row['image']}")
        prediction = dict(row)
        try:
            result = actors.judge([Path(row["image"])], "", "", config.seed, binary=True)
            prediction.update(prediction=result["labels"][0], reason=result["reason"])
        except Exception as exc:
            prediction.update(prediction=None, error=f"{type(exc).__name__}: {exc}")
        prediction["trace"] = actors.trace
        report["rows"].append(prediction)
        save_json(output, report)
        print(f"Validazione: {len(report['rows'])}/{sample}", flush=True)
    report["metrics"] = classification_report(report["rows"], config.seed)
    report["note"] = "Closed-set binario; valutare accuratezza, kappa e sottogruppi prima del run. Nessuna soglia arbitraria imposta."
    save_json(output, report)
    return report


def check_validation(config, path, provenance):
    validation = read_json(path)
    identity = validation["identity"]
    if identity.get("prompts_sha256") != fingerprint([JUDGE, VALIDATION_JUDGE]):
        raise ValueError("I prompt del giudice sono cambiati dopo la validazione")
    if identity["config"]["mock"] or identity["config"]["judge"] != config.judge:
        raise ValueError("Serve una validazione reale dello stesso modello giudice")
    old_digest = identity["provenance"].get("ollama_digests", {}).get(config.judge)
    if old_digest != provenance.get("ollama_digests", {}).get(config.judge):
        raise ValueError("Pesi del giudice diversi da quelli validati")
    if "metrics" not in validation or validation["metrics"]["classified"] == 0:
        raise ValueError("Validazione incompleta o senza classificazioni utilizzabili")
    return sha256(path)
