# Pacchetto operativo Lium

Prima del noleggio, controllare il report locale `runs/judge-validation.json` e verificare saldo, disco e tariffa. Il pacchetto contiene solo sorgenti, preset, test e liste di professioni. Il pilot e il run principale hanno output distinti. Non archiviare token o cache dei pesi insieme ai risultati.

Pilot soltanto: una A6000 a massimo $0,42/h, TTL di 2 ore e budget di $1.
Il TTL corrisponde a $0,84 di calcolo a questa tariffa; verificare eventuali costi
accessori nel preventivo. Download e setup consumano questo tempo: se non basta,
recuperare gli artefatti e rivalutare, senza estendere automaticamente il noleggio.
Sostituire NODE con un identificativo verificato e POD con il nome restituito.
La scadenza elimina anche i dati locali: recuperare i risultati prima del termine.

Preparare il pacchetto dalla root del repository dopo i test:

```bash
mkdir -p runs/lium-flux-comparator
git archive --format=tar.gz -o runs/lium-flux-comparator/source.tar.gz HEAD ouroboros configs tests data/professions.txt data/provenance.json data/subsets requirements-models.txt pyproject.toml
```

```bash
source .venv/bin/activate
lium quote NODE --hours 2 --gpus 1 --format json
lium up NODE --name ouroboros-flux --template_id cbc0ffdc-aa35-441a-abaa-170a87c51a64 \
  --ttl 2h --budget 1 --no-backup --timeout 180 --no-ssh --yes --json
lium exec POD 'df -h /workspace; nvidia-smi'
lium scp POD runs/lium-flux-comparator/source.tar.gz /workspace/source.tar.gz
lium scp POD scripts/lium/run.sh /workspace/run.sh
lium exec POD --script scripts/lium/setup.sh --detach --log /workspace/setup.log
```

Attendere `/workspace/artifacts/setup.complete`. Fornire l'accesso temporaneo ai pesi FLUX come nel pilot precedente e rimuoverlo prima di archiviare o chiudere il pod.

```bash
lium exec POD 'bash /workspace/run.sh pilot' --detach --log /workspace/pilot.log
lium exec POD 'tail -20 /workspace/pilot.log; cat /workspace/artifacts/pilot.time'
```

Il pilot deve produrre 320 immagini verificate, 40 batch giudicati e una coppia primaria. Lo script separa il preflight con download dei pesi dal tempo di esecuzione del pilot. Stimare il costo finale come spesa già maturata + 30 × costo del pilot a pesi in cache + margine per rallentamenti e recupero dati. Se il tempo previsto supera la scadenza o il costo eccede il saldo, salvare gli artefatti e chiudere il pod.

Non avviare la run principale con il limite del pilot. Prima rivalutare costo,
saldo e durata, e definire separatamente i limiti del noleggio.

Per la run principale il controllo richiede solo identità del giudice e metriche. Preparare una copia ridotta del report per il pod, conservando il report completo in locale:

```bash
python - <<'PY'
import json
from pathlib import Path
src = json.loads(Path("runs/judge-validation.json").read_text())
assert src["metrics"]["classified"] > 0
Path("runs/lium-flux-comparator/judge-validation.json").write_text(
    json.dumps({k: src[k] for k in ("identity", "metrics")}, indent=2) + "\n"
)
PY
lium scp POD runs/lium-flux-comparator/judge-validation.json /workspace/judge-validation.json
lium exec POD 'bash /workspace/run.sh main' --detach --log /workspace/main.log
```

Controllare il log e lo spazio a intervalli regolari. Salvare periodicamente i risultati in locale. Per un archivio coerente fare lo snapshot del checkpoint atomico prima di copiare le immagini da esso referenziate, oppure a processo fermo. A fine run scaricare output e log, verificare gli hash e chiudere il pod con `lium rm POD --yes --format json`. Usare `lium record ID --format json` per il costo registrato e `lium ps` per verificare la chiusura.
