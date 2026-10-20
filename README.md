# Ouroboros

Implementazione Python dell'esperimento di bias di genere della tesi, con ricerca e verifica indipendente. Il progetto separa il ciclo sperimentale dai modelli e dall'analisi; non usa framework per agenti, database o servizi a pagamento.

**Modelli scelti**

| Ruolo | Modello | Esecuzione |
|---|---|---|
| Target 1 | `black-forest-labs/FLUX.2-klein-4B` | Diffusers, 4 passi, guidance 1 |
| Target 2 | `Qwen/Qwen-Image-2.1` | Diffusers, 40 passi |
| Attaccante | `ministral-3:8b` | Ollama, temperatura 1 |
| Giudice | `qwen3-vl:8b-instruct-q4_K_M` | Ollama, temperatura 0 |
| Seconda misura | FairFace ResNet-34 | PyTorch + dlib, post hoc |

L'attaccante usa Ministral 3 8B e il giudice Qwen3-VL 8B, appartenenti a famiglie diverse. L'adeguatezza va verificata nel pilot; una famiglia diversa non garantisce indipendenza degli errori. I nomi sono configurabili. L'attaccante riceve solo testo e feedback della ricerca.

## Installazione

Usare Python 3.11 o successivo, preferibilmente in un ambiente dedicato:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Questo basta per test, mock e statistiche principali. Per l'inferenza installare PyTorch adatto alla propria GPU seguendo [le istruzioni ufficiali](https://pytorch.org/get-started/locally/), quindi:

```bash
pip install -e '.[models]'
pip install -r requirements-models.txt
ollama pull ministral-3:8b
ollama pull qwen3-vl:8b-instruct-q4_K_M
```

Ollama deve essere in esecuzione (`ollama serve`, se non già avviato dall'app). La versione deve supportare Qwen3-VL e il parametro `think: false`. Non vengono scaricati automaticamente modelli Ollama; Diffusers scarica i pesi del target al primo utilizzo. Qwen Image **2.1** usa `QwenImage21Pipeline`, non la pipeline della versione precedente. La revisione Diffusers è fissata in `requirements-models.txt`; dopo aver preparato l'ambiente salvare anche `pip freeze > requirements-lock.txt`.

I preset usano la risoluzione documentata per ciascun modello: 1024² per FLUX, 2048² per Qwen. Il confronto controlla il numero di immagini, **non** secondi GPU o FLOP. Modificare larghezza/altezza prima del run per un disegno a risoluzione comune, dichiarando la scelta.

`device: auto` seleziona CUDA, poi MPS, poi CPU. Su CUDA è attivo il CPU offload; su MPS il modello viene spostato sul device senza quel meccanismo. MPS e CPU sono percorsi implementati ma non verificati con i pesi reali in questa sessione. L'esecuzione sequenziale scarica i modelli fra le fasi: riduce il picco, aumenta i tempi di ricaricamento. Per un controllo statico impostare `available_memory_gb` e una stima conservativa `target_estimate_gb` nel JSON: non sono stime automatiche di VRAM e RAM. Con il valore predefinito 0 il controllo è disattivato.

## Verifica rapida senza GPU

```bash
python -m unittest discover -s tests -v
python -m ouroboros run --config configs/flux-test.json --mock --output runs/mock
```

Il comando stampa la directory del run, per esempio `runs/mock/<id>`. Passare quel percorso ai comandi successivi:

```bash
python -m ouroboros report runs/mock/<id>
python -m ouroboros replay runs/mock/<id> --limit 8
```

Il mock produce immagini a tinta unita e label artificiali. Serve esclusivamente a verificare il software; i suoi report sono marcati **MOCK** e non sono risultati della tesi.

## Esecuzione su Modal

L’adattatore opzionale [modal_app.py](modal_app.py) esegue la CLI con GPU, Ollama e
volumi persistenti. La [guida Modal](docs/modal.md) descrive autenticazione, piano
locale senza cloud, pilot, validazione, recupero risultati e limiti di spesa.

## 1. Validazione del giudice

Scaricare dal [dataset ufficiale T2ISafety](https://huggingface.co/datasets/OpenSafetyLab/t2i_safety_dataset/tree/main) `hf_test_fairness_generated.json` e `test.zip`, quindi estrarre lo ZIP. Il test set comprende anche file per immagini reali; il preset seguente usa solo quelle generate. Non usare le caption come etichette e non miscelare train e test.

```bash
hf download OpenSafetyLab/t2i_safety_dataset \
  hf_test_fairness_generated.json test.zip \
  --repo-type dataset --local-dir data/t2isafety
unzip data/t2isafety/test.zip -d data/t2isafety
python -m ouroboros import-t2isafety \
  --annotations data/t2isafety/hf_test_fairness_generated.json \
  --image-root data/t2isafety \
  --output data/t2isafety/manifest.json
python -m ouroboros validate-judge \
  --config configs/flux-test.json \
  --manifest data/t2isafety/manifest.json --sample 150 \
  --output runs/judge-validation.json
```

`--image-root` deve contenere la cartella `evaluator_test` citata nelle annotazioni; se lo ZIP aggiunge una directory esterna, adeguare il percorso. L'importatore controlla ogni file. Legge solo le annotazioni `Gender`, `Age`, `Race` del dataset: nessuna annotazione viene inventata. Il campione è casuale e riproducibile, non stratificato; i sottogruppi piccoli vanno interpretati con cautela.

La validazione restituisce accuracy, Cohen's kappa, matrice di confusione, errori e accuracy per età/razza con bootstrap. È closed-set binaria, come il dataset; in produzione il giudice può usare `unclear`. Il file permette di riprendere il pilot senza ripetere le immagini già processate. Le classificazioni fallite rimangono errori visibili; per riprovare il pilot usare un nuovo output.

Esaminare i risultati prima dell'esperimento. Il codice verifica che esista una validazione dello stesso digest del giudice, **non certifica automaticamente** che l'accuratezza sia sufficiente: la tesi non prescrive una soglia numerica. Un run senza pilot richiede `--exploratory` ed è registrato come preliminare.

## 2. Esperimento appaiato

```bash
python -m ouroboros doctor --config configs/flux-test.json
python -m ouroboros run --config configs/flux-test.json \
  --validation runs/judge-validation.json
```

Dopo il test, usare i preset full per ciascun target:

```bash
python -m ouroboros run --config configs/flux-full.json \
  --validation runs/judge-validation.json
python -m ouroboros run --config configs/qwen-full.json \
  --validation runs/judge-validation.json
```

| Modalità | Professioni | Immagini/batch (M) | Ricerca (B) | Verifica per prompt (V) | Immagini totali/target |
|---|---:|---:|---:|---:|---:|
| test | 10 | 2 | 5 | 5 | 300 |
| full | 175 | 8 | 5 | 5 | 21.000 |

Per ogni professione:

1. La ricerca prova B prompt, senza arresto anticipato, guidata da `abs(n_female - n_male) / M`. Le immagini unclear restano nel denominatore.
2. Si congela il prompt con punteggio maggiore; a parità vince il primo. Batch tutti unclear ed errori non possono vincere.
3. Il prompt scelto e il prompt neutro generano ciascuno V*M immagini nuove, con semi appaiati tra condizioni e distinti dalla ricerca. Il giudice vede solo le immagini. Il feedback di verifica non torna all'attaccante.
4. Il report aggrega tutte le immagini della verifica e confronta quota femminile, skew assoluto e coverage. Non usa il massimo dei batch o una soglia di successo.

Nel JSON, `iterations` è B e `verification_batches` è V. `max_generations` limita le chiamate al target, incluse quelle interrotte o fallite. Il piano è `professioni * M * (B + 2*V)`: la ricerca è un costo aggiuntivo rispetto alla verifica appaiata. I valori full sono un punto di partenza da dimensionare con un pilot su GPU, non una garanzia di rientrare nel budget.

Ripetere lo stesso comando riprende il checkpoint. La selezione viene salvata prima della verifica e non viene ricalcolata usando il holdout. La configurazione e i digest dei modelli identificano il run; gli hash del codice e delle immagini bloccano una ripresa incompatibile. Usare un solo processo per directory di run. Una chiamata interrotta è conteggiata anche se non salva un'immagine: prevedere margine nel tetto iniziale.

Prima di iniziare una professione si verifica il budget per completare ricerca e verifica. Se manca, il run si ferma. Errori e rifiuti consumano lo slot; se nessun candidato è selezionabile si salta la verifica e lo si registra. Il confronto usa solo coppie complete e classificabili, esponendo tutte le esclusioni. I vecchi JSON con `quorum` o `baseline` non sono riutilizzabili: per quei run usare `main`.

## 3. FairFace e report

```bash
pip install -e '.[fairface]'
```

Scaricare dal [repository ufficiale FairFace](https://github.com/dchen236/FairFace) il checkpoint **7-race** `fairface_alldata_20191111.pt` e i due file dlib `mmod_human_face_detector.dat` e `shape_predictor_5_face_landmarks.dat` (decomprimerli se necessario). Non usare il checkpoint 4-race: le posizioni delle teste sono diverse. Per compilare dlib possono essere necessari CMake e un compilatore C++.

```bash
python -m ouroboros fairface runs/<id> \
  --weights models/fairface_alldata_20191111.pt \
  --detector models/mmod_human_face_detector.dat \
  --landmarks models/shape_predictor_5_face_landmarks.dat
python -m ouroboros report runs/<id>
python -m ouroboros replay runs/<id> --limit 8
```

FairFace è applicato a tutte le immagini salvate, senza entrare nel feedback dell'attaccante. Classifica ogni volto con l'allineamento dlib; conserva il numero di volti e restituisce solo il genere percepito. Il kappa per immagine usa esattamente un volto; zero volti non viene convertito in `male`. I file dei pesi e delle immagini sono identificati da hash.

L'analisi opzionale delle strategie usa MiniLM + HDBSCAN:

```bash
pip install -e '.[analysis]'
python -m ouroboros report runs/<id> --cluster-strategies
```

Output principali: `checkpoint.json` (storia, scelta congelata e risposte grezze), `images/`, `report.md`, `report.json`, `per_profession.csv`; dopo i comandi specifici anche `fairface.json`, `strategies.json`, `replay.json`. I report contengono conteggi, quote e skew aggregati sulla verifica, differenze appaiate e bootstrap 95% sulle professioni. Le quote per professione includono IC Wilson. Le metriche non definite sono `null`; gli errori non diventano zeri. FairFace confronta solo immagini di verifica nel report.

## Come leggere il codice

- `config.py`: parametri e identificazione del run.
- `experiment.py`: il ciclo principale e la memoria.
- `models.py`, `prompts.py`: accesso ai modelli e prompt versionati.
- `storage.py`: checkpoint e hash.
- `validation.py`, `fairface.py`: i due controlli del giudice.
- `analysis.py`, `replay.py`: metriche, strategie e riproducibilità.
- `cli.py`: comandi da terminale.

Le scelte metodologiche e le differenze rispetto alla formulazione della tesi sono descritte in [docs/protocol.md](docs/protocol.md). La provenienza dei 175 nomi è in `data/provenance.json`: l'elenco è ricavato dai valori effettivi della colonna `profession`, non dalla lista incompleta nella dataset card.

## Riferimenti verificati

- [FLUX.2-klein-4B: model card e inferenza](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B).
- [Qwen Image 2.1: model card e inferenza](https://huggingface.co/Qwen/Qwen-Image-2.1).
- [Ministral 3 8B Q4](https://ollama.com/library/ministral-3:8b), [Qwen3-VL-8B Instruct Q4](https://ollama.com/library/qwen3-vl:8b-instruct-q4_K_M).
- [Stable Bias](https://huggingface.co/datasets/stable-bias/professions), licenza CC-BY-SA-4.0 per i dati derivati; [T2ISafety](https://github.com/adwardlee/t2i_safety); [FairFace](https://github.com/dchen236/FairFace).
