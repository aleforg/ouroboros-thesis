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


def classification_report(rows, seed=42):
    valid = [r for r in rows if r.get("prediction") is not None]
    pairs = [(r["label"], r["prediction"]) for r in valid]
    counts = Counter(pairs)
    confusion = {a: {b: counts[a, b]
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


def verification_summary(records, batches, images):
    labels = [label for r in records if r['status'] == 'judged' for label in r['labels']]
    female, male = labels.count('female'), labels.count('male')
    classified = female + male
    fraction = female / classified if classified else None
    complete = (len(records) == batches and all(r['status'] == 'judged' and
                len(r['images']) == len(r['labels']) == images for r in records))
    return {'complete': complete, 'generated': sum(len(r['images']) for r in records),
            'judged': len(labels), 'female': female, 'male': male,
            'unclear': labels.count('unclear'), 'female_ratio': fraction,
            'absolute_skew': abs(2 * fraction - 1) if fraction is not None else None,
            'coverage': classified / len(labels) if labels else None,
            'female_ratio_ci95': wilson(female, classified)}


def wilson(successes, total):
    if not total:
        return None
    z = 1.959963984540054
    p = successes / total
    denominator = 1 + z*z/total
    center = (p + z*z/(2*total)) / denominator
    radius = z * math.sqrt(p*(1-p)/total + z*z/(4*total*total)) / denominator
    return [max(0, center-radius), min(1, center+radius)]


def make_report(root):
    root = Path(root)
    state = read_json(root / 'checkpoint.json')
    config = state['config']
    if config.get('protocol') != 'search-verification-v1':
        raise ValueError('Report del vecchio protocollo: usare il branch main')
    summaries, rows = {}, []
    for profession in state['professions']:
        summaries[profession] = {}
        for condition in ('selected', 'neutral'):
            records = [r for r in state['records'] if r['profession'] == profession
                       and r['condition'] == condition]
            summary = verification_summary(records, config['verification_batches'], config['images'])
            summaries[profession][condition] = summary
            rows.append({'profession': profession, 'condition': condition, **summary})
    # Le due medie e i delta usano esattamente le stesse professioni.
    complete = [v for p, v in summaries.items() if p in state['completed']
                and all(v[c]['complete'] for c in ('selected', 'neutral'))]
    paired = [v for v in complete if all(v[c]['female_ratio'] is not None for c in ('selected', 'neutral'))]
    seed = config['seed']
    conditions = {c: {m: bootstrap([v[c][m] for v in paired], seed=seed)
                     for m in ('female_ratio', 'absolute_skew', 'coverage')}
                  for c in ('selected', 'neutral')}
    comparison = {'n_pairs': len(paired), 'complete_pairs': len(complete),
                  'excluded_completed_pairs': len(state['completed']) - len(paired)}
    for metric in ('female_ratio', 'absolute_skew', 'coverage'):
        sample = complete if metric == 'coverage' else paired
        comparison['delta_' + metric] = bootstrap(
            [v['selected'][metric] - v['neutral'][metric] for v in sample], seed=seed)
    report = {'protocol': config['protocol'], 'mock': config['mock'],
              'finished': state['finished'], 'stop_reason': state['stop_reason'],
              'professions_total': len(summaries), 'professions_complete': len(state['completed']),
              'generation_calls': state['generation_calls'],
              'search_batches': config['iterations'],
              'verification_images_per_condition': config['verification_batches'] * config['images'],
              'statuses': dict(Counter(r['status'] for r in state['records'])),
              'no_candidate': [p for p, choice in state['selections'].items() if choice is None],
              'selections': state['selections'], 'conditions': conditions, 'paired': comparison,
              'per_profession': summaries}
    if (root / 'fairface.json').exists():
        report['fairface'] = convergence(state, read_json(root / 'fairface.json'))
    report['notes'] = [
        'MOCK: nessun risultato scientifico' if config['mock'] else 'Genere percepito, non identità reali',
        'Ricerca esclusa dalle stime finali; scelta congelata prima della verifica',
        'Verifica: etichette aggregate su tutti i batch, mai massimo o media dello skew dei batch',
        'Quota femminile e skew calcolati sulle sole immagini classificabili; coverage riportata separatamente',
        'Confronto sulle coppie complete con almeno una immagine classificabile in ciascuna condizione',
        'Delta coverage include anche le coppie complete tutte unclear; nessun errore imputato a zero',
        'IC bootstrap percentile: 2000 ricampionamenti appaiati di professioni; con una professione non stimano variabilità tra professioni',
        'IC Wilson della quota per professione: incertezza binomiale condizionata alle immagini classificabili, non errore del giudice',
        'Lo skew assoluto è una stima campionaria, può essere positivo anche per un generatore bilanciato',
        'Parità di immagini nella verifica; il costo della ricerca è aggiuntivo e registrato',
        'Il confronto misura ricerca e riscrittura insieme; non isola il contributo del feedback',
    ]
    save_json(root / 'report.json', report)
    with (root / 'per_profession.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    def display(item):
        if item['estimate'] is None:
            return 'non definito'
        low, high = item['ci95']
        return f"{item['estimate']:.3f} [{low:.3f}, {high:.3f}]"
    title = 'Ouroboros — MOCK, non usare nella tesi' if config['mock'] else 'Ouroboros — verifica'
    lines = [f'# {title}', '', f"Coppie utilizzabili: {len(paired)}/{len(summaries)}.", '',
             '| Condizione | Quota femminile (IC 95%) | Skew assoluto (IC 95%) | Coverage (IC 95%) |',
             '|---|---|---|---|']
    for condition, values in conditions.items():
        lines.append('| ' + condition + ' | ' + ' | '.join(display(values[m]) for m in
                     ('female_ratio', 'absolute_skew', 'coverage')) + ' |')
    lines += ['', 'Δ skew (selected − neutral): ' + display(comparison['delta_absolute_skew']), '',
              f"Chiamate di generazione, ricerca inclusa: {state['generation_calls']}.", '']
    lines += ['- ' + note for note in report['notes']]
    (root / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return report


def convergence(state, fairface):
    """Confronto esplorativo con FairFace sulle sole immagini di verifica."""
    lookup = {r['path']: r for r in fairface['images']}
    config = state['config']
    clusters, deltas = [], []
    coverage = Counter()
    for profession in state['completed']:
        ratios, pairs = {}, []
        for condition in ('selected', 'neutral'):
            records = [r for r in state['records'] if r['profession'] == profession and r['condition'] == condition]
            if not verification_summary(records, config['verification_batches'], config['images'])['complete']:
                continue
            faces, complete = [], True
            for record in records:
                for image, label in zip(record['images'], record['labels']):
                    ff = lookup.get(image['path'])
                    if ff is None or ff.get('error') or ff.get('sha256') != image['sha256']:
                        coverage['missing_error_or_hash_mismatch'] += 1
                        complete = False
                        continue
                    labels = ff['labels']
                    coverage['images'] += 1
                    coverage['no_face'] += not labels
                    coverage['multiple_faces'] += len(labels) > 1
                    faces.extend(labels)
                    if len(labels) == 1:
                        pairs.append((label, labels[0]))
            if complete and faces:
                ratios[condition] = faces.count('female') / len(faces)
        if pairs:
            clusters.append(pairs)
        if len(ratios) == 2:
            deltas.append(abs(2*ratios['selected']-1) - abs(2*ratios['neutral']-1))
    return {'coverage': dict(coverage), 'exploratory': True,
            'kappa_single_face': bootstrap(clusters, lambda groups: kappa([p for group in groups for p in group]), config['seed']),
            'delta_absolute_skew': bootstrap(deltas, seed=config['seed']),
            'note': 'FairFace conta i volti rilevati, il giudice classifica immagini: denominatori diversi'}


def cluster_strategies(root):
    from sentence_transformers import SentenceTransformer
    from sklearn.cluster import HDBSCAN

    root = Path(root)
    state = read_json(root / "checkpoint.json")
    records = [r for r in state["records"] if r["condition"] == "search" and r["status"] in INFORMATIVE]
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
        groups = [[r["scores"]["search_score"] for r in selected if r["profession"] == p]
                  for p in sorted({r["profession"] for r in selected})]
        results[family] = {"iterations": len(selected), "search_score": bootstrap(
            groups, lambda sample: np.mean([v for group in sample for v in group]), state["config"]["seed"])}
    save_json(root / "strategies.json", {"mapping": mapping, "families": results, "exploratory": True})
