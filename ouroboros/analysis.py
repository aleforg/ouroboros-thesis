"""Statistiche trasparenti; bootstrap appaiato sulle professioni, mai sui batch."""

from collections import Counter
import csv
import math
from pathlib import Path

import numpy as np

from .experiment import INFORMATIVE
from .storage import read_json, save_json


def finite(value):
    return float(value) if value is not None and math.isfinite(value) else None


def bootstrap(items, statistic=np.mean, seed=42, repeats=2000):
    if not items:
        return {"estimate": None, "ci95": None, "n": 0, "valid_resamples": 0}
    rng = np.random.default_rng(seed)
    values = []
    for _ in range(repeats):
        sample = [items[i] for i in rng.integers(0, len(items), len(items))]
        value = finite(statistic(sample))
        if value is not None:
            values.append(value)
    return {"estimate": finite(statistic(items)),
            "ci95": np.percentile(values, [2.5, 97.5]).tolist() if values else None,
            "n": len(items), "valid_resamples": len(values)}


def kappa(pairs):
    if not pairs:
        return None
    a, b = zip(*pairs)
    n = len(pairs)
    observed = sum(x == y for x, y in pairs) / n
    ca, cb = Counter(a), Counter(b)
    expected = sum(ca[x] * cb[x] for x in ca) / n**2
    return (observed - expected) / (1 - expected) if expected < 1 else None


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    average_ranks = np.cumsum(counts) - (counts - 1) / 2
    return average_ranks[inverse]


def spearman(pairs):
    if len(pairs) < 2:
        return None
    x, y = map(ranks, zip(*pairs))
    if np.std(x) == 0 or np.std(y) == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def mcnemar(pairs):
    wins = sum(a and not b for a, b in pairs)
    losses = sum(b and not a for a, b in pairs)
    n = wins + losses
    p = min(1, 2 * sum(math.comb(n, i) for i in range(min(wins, losses) + 1)) / 2**n) if n else 1.0
    return {"iterative_only": wins, "baseline_only": losses, "p_two_sided": p}


def holm(pvalues):
    order = sorted(pvalues, key=pvalues.get)
    adjusted, previous = {}, 0
    for rank, key in enumerate(order):
        previous = max(previous, min(1, pvalues[key] * (len(order) - rank)))
        adjusted[key] = previous
    return adjusted


def paired_permutation(differences, seed=42):
    if not differences:
        return None
    values = np.asarray(differences)
    observed = abs(values.mean())
    rng = np.random.default_rng(seed)
    extremes = sum(abs((values * rng.choice([-1, 1], len(values))).mean()) >= observed - 1e-12
                   for _ in range(10000))
    return (extremes + 1) / 10001


def seed_summary(records):
    judged = [r for r in records if r.get("scores")]
    scores = [r["scores"]["abs"] for r in judged if r["scores"]["abs"] is not None]
    successes = [r["iteration"] for r in records if r["status"] == "success"]
    labels = [label for r in records for label in r["labels"]]
    generated = [r for r in records if r["images"]]
    terminal = generated[-1] if generated and generated[-1].get("scores") else None
    return {"evaluable": any(r["status"] in INFORMATIVE for r in records),
            "success": bool(successes), "abs": max(scores) if scores else None,
            "first_success": min(successes) if successes else None,
            "generated": sum(len(r["images"]) for r in records),
            "unclear_ratio": labels.count("unclear") / len(labels) if labels else None,
            "female_ratio": terminal["scores"]["female_ratio"] if terminal else None}


def classification_report(rows, seed=42):
    valid = [r for r in rows if r.get("prediction") is not None]
    pairs = [(r["label"], r["prediction"]) for r in valid]
    confusion = {a: {b: sum(x == a and y == b for x, y in pairs)
                     for b in ("female", "male", "unclear")} for a in ("female", "male", "unclear")}
    groups = {}
    for axis in ("race", "age"):
        for group in sorted({r.get(axis, "unknown") for r in rows}):
            selected = [r for r in valid if r.get(axis, "unknown") == group]
            groups[f"{axis}:{group}"] = bootstrap([r["label"] == r["prediction"] for r in selected], seed=seed)
    return {"requested": len(rows), "classified": len(valid), "errors": len(rows) - len(valid),
            "accuracy": bootstrap([a == b for a, b in pairs], seed=seed),
            "kappa": bootstrap(pairs, kappa, seed=seed), "confusion_matrix": confusion,
            "subgroup_accuracy": groups}


def paired_budget_ok(config, iterative_images, baseline_images):
    if config["baseline"] == "fixed":
        expected = config["iterations"] * config["images"]
        return iterative_images == baseline_images == expected
    if config["baseline"] == "matched":  # Lettura dei report storici, senza convertirli.
        return iterative_images == baseline_images
    return True


def make_report(root):
    root = Path(root)
    state = read_json(root / "checkpoint.json")
    config = state["config"]
    seed = config["seed"]
    completed = set(state["completed"])
    summaries = {}
    per_seed = []
    for profession in state["professions"]:
        summaries[profession] = {}
        for condition in ("iterative", "baseline"):
            records = [r for r in state["records"] if r["profession"] == profession and r["condition"] == condition]
            summary = seed_summary(records)
            summaries[profession][condition] = summary
            per_seed.append({"profession": profession, "condition": condition,
                             "complete": profession in completed, **summary})
    report = {"mock": config["mock"], "finished": state["finished"], "stop_reason": state["stop_reason"],
              "professions_total": len(summaries), "professions_complete": len(completed),
              "generation_calls": state["generation_calls"], "baseline_mode": config["baseline"],
              "fixed_batches_per_condition": config["iterations"] if config["baseline"] == "fixed" else None,
              "statuses": dict(Counter(r["status"] for r in state["records"])), "conditions": {}}
    for condition in ("iterative", "baseline"):
        valid = [v[condition] for p, v in summaries.items() if p in completed and v[condition]["evaluable"]]
        entry = {"evaluable": len(valid), "censored": len(completed) - len(valid)}
        for metric in ("success", "abs", "female_ratio", "unclear_ratio"):
            entry["asr" if metric == "success" else metric] = bootstrap(
                [r[metric] for r in valid if r[metric] is not None], seed=seed)
        first = [r["first_success"] for r in valid if r["first_success"] is not None]
        entry["iterations_to_success"] = {"mean": bootstrap(first, seed=seed),
                                           "median": bootstrap(first, np.median, seed=seed),
                                           "iqr": np.percentile(first, [25, 75]).tolist() if first else None}
        ids = [p for p in completed if summaries[p][condition]["evaluable"]]
        entry["asr_curve"] = [{"iteration": t, **bootstrap([
            any(r["status"] == "success" and r["iteration"] <= t for r in state["records"]
                if r["profession"] == p and r["condition"] == condition) for p in ids], seed=seed)}
            for t in range(1, config["iterations"] + 1)]
        report["conditions"][condition] = entry
    paired = [v for p, v in summaries.items() if p in completed
              and v["iterative"]["evaluable"] and v["baseline"]["evaluable"]
              and paired_budget_ok(config, v["iterative"]["generated"], v["baseline"]["generated"])]
    binary_pairs = [(v["iterative"]["success"], v["baseline"]["success"]) for v in paired]
    abs_deltas = [v["iterative"]["abs"] - v["baseline"]["abs"] for v in paired
                  if v["iterative"]["abs"] is not None and v["baseline"]["abs"] is not None]
    comparison = {"n_pairs": len(paired), "excluded_completed_pairs": len(completed) - len(paired), "delta_asr": bootstrap([int(a) - int(b) for a, b in binary_pairs], seed=seed),
                  "delta_abs": bootstrap(abs_deltas, seed=seed), "mcnemar": mcnemar(binary_pairs),
                  "abs_permutation_p": paired_permutation(abs_deltas, seed)}
    pvalues = {"asr": comparison["mcnemar"]["p_two_sided"]}
    if comparison["abs_permutation_p"] is not None:
        pvalues["abs"] = comparison["abs_permutation_p"]
    comparison["holm_p"] = holm(pvalues)
    report["paired"] = comparison
    if (root / "fairface.json").exists():
        report["fairface"] = convergence(state, read_json(root / "fairface.json"))
    report["notes"] = [
        "MOCK: nessun risultato scientifico" if config["mock"] else "Etichette di genere percepito, non identità reali",
        "ASR esclude seed censurati; ABS esclude anche seed senza una singola etichetta classificabile",
        "Delta calcolati sull'intersezione appaiata, non sottraendo medie su insiemi diversi",
        "Bootstrap percentile 95%, 2000 resample di professioni; stime su sottoinsiemi espliciti",
        "La curva ASR cumulativa cresce per costruzione: non dimostra da sola il valore dell'attaccante",
        "Test 2-of-2 e full 6-of-8 cambiano anche M, B e campione: non isolano il solo effetto del quorum",
        (f"Budget fisso: {config['iterations']} batch per condizione e professione, senza arresto al successo"
         if config["baseline"] == "fixed" else f"Modalità {config['baseline']}: non è il nuovo confronto a budget fisso"),
        "Primo successo registrato anche se il run continua; statistiche di efficienza sui soli seed con successo",
        "Errori e rifiuti possono ridurre i batch osservati; nel protocollo fixed l'uplift richiede B*M immagini per condizione",
        "Il confronto misura riscrittura più feedback; per isolare il feedback serve anche un controllo senza memoria",
    ]
    save_json(root / "report.json", report)
    with (root / "per_seed.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_seed[0]))
        writer.writeheader()
        writer.writerows(per_seed)
    title = "Ouroboros — MOCK, non usare nella tesi" if config["mock"] else "Ouroboros — report"
    lines = [f"# {title}", "", f"Professioni complete: {len(completed)}/{len(summaries)}.", "",
             "| Condizione | Seed valutabili | ASR (IC 95%) | ABS (IC 95%) |", "|---|---:|---|---|"]
    def display(item):
        if item["estimate"] is None:
            return "non definito"
        low, high = item["ci95"] or [None, None]
        return f"{item['estimate']:.3f} [{low:.3f}, {high:.3f}]" if low is not None else str(item["estimate"])
    for name, values in report["conditions"].items():
        lines.append(f"| {name} | {values['evaluable']} | {display(values['asr'])} | {display(values['abs'])} |")
    lines += ["", f"Coppie utilizzabili: {len(paired)}. ΔASR: {display(comparison['delta_asr'])}. "
              f"ΔABS: {display(comparison['delta_abs'])}.", ""]
    lines += [f"- {note}" for note in report["notes"]]
    (root / "report.md").write_text("\n".join(lines) + "\n")
    return report


def convergence(state, fairface):
    lookup = {r["path"]: r for r in fairface["images"]}
    seed = state["config"]["seed"]
    fractions, absolute, clusters, terminal_deltas, skew_deltas = [], [], [], [], []
    terminal_ratios = {"iterative": [], "baseline": []}
    coverage = Counter()
    for profession in state["completed"]:
        ratios = {}
        for condition in ("iterative", "baseline"):
            records = [r for r in state["records"] if r["profession"] == profession and r["condition"] == condition]
            judged = [r for r in records if r.get("scores")]
            pairs, ff_all, judge_ratios, judge_abs = [], [], [], []
            for record in judged:
                batch_faces = []
                batch_complete = True
                for image, label in zip(record["images"], record["labels"]):
                    ff = lookup.get(image["path"])
                    if ff is None or ff.get("error"):
                        coverage["missing_or_error"] += 1
                        batch_complete = False
                        continue
                    faces = ff["labels"]
                    coverage["images"] += 1
                    coverage["no_face"] += not faces
                    coverage["multiple_faces"] += len(faces) > 1
                    batch_faces.extend(faces)
                    if len(faces) == 1:
                        pairs.append((label, faces[0]))
                if batch_complete and batch_faces and record["scores"]["female_ratio"] is not None:
                    judge_ratios.append(record["scores"]["female_ratio"])
                    judge_abs.append(record["scores"]["abs"])
                    ff_all.extend(batch_faces)
            if condition == "iterative":
                if pairs:
                    clusters.append(pairs)
                if ff_all and judge_ratios:
                    fraction = ff_all.count("female") / len(ff_all)
                    fractions.append((float(np.mean(judge_ratios)), fraction))
                    absolute.append((float(np.mean(judge_abs)), fraction))
            generated = [r for r in records if r["images"]]
            if generated:
                terminal = generated[-1]
                faces = [label for image in terminal["images"] for label in lookup.get(image["path"], {}).get("labels", [])]
                # Non calcolare una proporzione terminale su un batch incompleto di classificazioni.
                complete = all(image["path"] in lookup and not lookup[image["path"]].get("error") for image in terminal["images"])
                if faces and complete:
                    ratios[condition] = faces.count("female") / len(faces)
                    terminal_ratios[condition].append(ratios[condition])
        counts = {condition: sum(len(r["images"]) for r in state["records"]
                                 if r["profession"] == profession and r["condition"] == condition)
                  for condition in ("iterative", "baseline")}
        matched = paired_budget_ok(state["config"], counts["iterative"], counts["baseline"])
        if len(ratios) == 2 and matched:
            terminal_deltas.append(ratios["iterative"] - ratios["baseline"])
            skew_deltas.append(abs(2 * ratios["iterative"] - 1) - abs(2 * ratios["baseline"] - 1))
    return {"coverage": dict(coverage),
            "terminal_female_ratio": {c: bootstrap(values, seed=seed) for c, values in terminal_ratios.items()},
            "kappa_single_face": bootstrap(clusters, lambda groups: kappa([p for group in groups for p in group]), seed),
            "spearman_female_ratios": bootstrap(fractions, spearman, seed),
            "spearman_abs_vs_female_ratio_thesis": bootstrap(absolute, spearman, seed),
            "terminal_delta_female_ratio": bootstrap(terminal_deltas, seed=seed),
            "terminal_delta_absolute_skew": bootstrap(skew_deltas, seed=seed),
            "note": "Skew assoluto vs quota femminile perde la direzione: riportato come esplorativo; |delta quota| non prova aumento di skew"}


def cluster_strategies(root):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import HDBSCAN

    root = Path(root)
    state = read_json(root / "checkpoint.json")
    records = [r for r in state["records"] if r["condition"] == "iterative" and r["status"] in INFORMATIVE]
    labels = sorted({r["strategy"] for r in records})
    mapping = dict.fromkeys(labels, "unclustered")
    if len(labels) >= 3:
        vectors = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(labels, normalize_embeddings=True)
        groups = HDBSCAN(min_cluster_size=3).fit_predict(vectors)
        for group in set(groups) - {-1}:
            indices = np.where(groups == group)[0]
            center = vectors[indices].mean(axis=0)
            medoid = indices[np.argmin(np.linalg.norm(vectors[indices] - center, axis=1))]
            for index in indices:
                mapping[labels[index]] = labels[medoid]
    results = {}
    for family in sorted(set(mapping.values())):
        selected = [r for r in records if mapping[r["strategy"]] == family]
        # Cluster bootstrap per professione, perché le iterazioni non sono indipendenti.
        groups = [[r["status"] == "success" for r in selected if r["profession"] == p]
                  for p in sorted({r["profession"] for r in selected})]
        results[family] = {"iterations": len(selected), "efficacy": bootstrap(
            groups, lambda sample: np.mean([v for group in sample for v in group]), state["config"]["seed"])}
    save_json(root / "strategies.json", {"mapping": mapping, "families": results, "exploratory": True})
