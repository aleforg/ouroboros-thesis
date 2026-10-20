# Eseguire Ouroboros su Modal

L'adattatore `modal_app.py` esegue la CLI esistente in un container Linux Python 3.11,
con Diffusers e Ollama sulla stessa GPU. Non modifica protocollo, preset o target:
Qwen rimane `Qwen/Qwen-Image-2.1`, non Turbo. I preset budget20 attuali sono ancora
M=4, B=3, V=5: 1.560 immagini per target con 30 professioni.

## Installazione e account

Dalla radice del repository, con l'ambiente virtuale attivo:

```bash
pip install -e '.[cloud]'
modal setup
```

`modal setup` apre l'autenticazione personale nel browser. Non inserire token nel
repository. Nel pannello Billing di Modal verificare credito disponibile, metodo
di pagamento richiesto per le GPU e impostare uno **spend limit** appropriato
(per esempio $0 di spesa aggiuntiva rispetto ai crediti). Un budget di avviso e un
limite di durata non costituiscono un limite monetario. CPU, RAM, avvio e download
nel container GPU consumano credito oltre all'inferenza.

Se Hugging Face richiede un token, creare un Secret Modal `huggingface` contenente
`HF_TOKEN` e impostare `OUROBOROS_HF_SECRET=huggingface` prima del comando. Questo
non sostituisce l'accettazione di eventuali condizioni del modello su Hugging Face.

## Controllo locale, senza cloud

```bash
python modal_app.py --config configs/flux-budget20.json \
  --professions data/subsets/budget20-30.txt --limit 30
python -m unittest discover -s tests -v
```

Il primo comando stampa il piano senza autenticazione, build o allocazione GPU.
Richiede solo l'SDK Modal e le dipendenze base. Nessun modello viene scaricato.

## Test infrastrutturale e pilot

Il test mock remoto usa solo CPU, salva immagini artificiali e verifica il volume:

```bash
modal run modal_app.py --action smoke
```

Il comando seguente **alloca una GPU**, avvia Ollama e scarica i due modelli degli
attori se mancanti; verifica le dipendenze ma non esegue inferenza sul target:

```bash
modal run modal_app.py --action doctor --config configs/flux-test.json
```

Pilot preliminare di una professione, senza validazione scientifica:

```bash
modal run --detach modal_app.py --action run \
  --config configs/flux-test.json --limit 1 --exploratory
```

`--detach` mantiene il job attivo se il terminale si disconnette. Il comando stampa
il link ai log Modal. Interrompere l'app dal pannello o con `modal app stop APP_ID`.
Non usare `modal deploy`: questo è un batch, non un endpoint permanente.

Il valore predefinito di `--limit` è **1**, anche con preset full. Specificarlo sempre
per il run definitivo. Il default di durata è un'ora inclusi download: un primo
avvio può impiegarne una parte importante e lasciare il pilot incompleto.

## Hardware e durata

```bash
OUROBOROS_GPU=A100-80GB OUROBOROS_HOURS=2 \
modal run --detach modal_app.py --action run \
  --config configs/qwen-test.json --limit 1 --exploratory
```

Default: L40S 48 GB, 2 core CPU, 64 GiB RAM. Il limite configurabile è massimo
23 ore, con 120 secondi aggiuntivi di timeout Modal per chiusura e commit.
Nessun retry automatico; massimo un container per funzione/app. **Non avviare più
app contemporaneamente sullo stesso run o sulla stessa cache**: questo limite non
è un lock globale tra invocazioni distinte di `modal run`.

Mantenere `sequential` e `cpu_offload` del preset finché il pilot non misura VRAM e
latenza. Caricare contemporaneamente target, attacker e judge può saturare 48 GB.
Gli 8 elementi di un batch sperimentale non implicano una singola inferenza GPU
vettorializzata: il ciclo esistente genera le immagini sequenzialmente.

## Persistenza, ripresa e risultati

- `ouroboros-models`, montato su `/cache`: cache Hugging Face e modelli Ollama.
- `ouroboros-results`, montato su `/results`: run, immagini, validazioni e ambiente.
- `/results/environment/<timestamp>`: comando, GPU, pacchetti e versione Ollama.

I pesi Ollama già presenti non vengono aggiornati automaticamente. Diffusers usa
la revisione del repository fissata in `requirements-models.txt`; Ollama viene
scaricato in fase di build dall'URL ufficiale corrente e la versione viene registrata.
Le altre dipendenze non sono completamente bloccate: conservare il manifest dei
pacchetti e l'immagine Modal del run per la riproducibilità.

Il wrapper effettua commit periodici e alla chiusura. Un arresto forzato del
container può perdere gli ultimi file non sincronizzati; non c'è una transazione
atomica tra tutti i PNG e il checkpoint. Prima di riprendere controllare i log.
Ripetere esattamente configurazione, elenco, limite, validazione e directory output;
il progetto verifica hash e identità. Cambiare `revision: main` quando il modello
upstream cambia può creare un nuovo run: per i run definitivi fissare lo SHA del
modello nella configurazione prima di iniziare. Il download conta nel tempo GPU.

```bash
modal volume ls ouroboros-results /runs
modal volume get ouroboros-results /runs/<id> runs/modal/<id>
python -m ouroboros report runs/modal/<id>
```

La cache dei pesi non è una copia dei risultati. I volumi persistono dopo la fine
del job; controllare anche eventuali costi di storage correnti.

## Validare il giudice sul cloud

Il manifest T2ISafety contiene percorsi assoluti. Non caricare senza modifica un
manifest costruito sul Mac: le immagini sul cloud sono in `/results/t2isafety`.
Caricare il dataset estratto nel volume:

```bash
modal volume create ouroboros-results
modal volume put ouroboros-results data/t2isafety /t2isafety
```

Se il volume esiste già, saltare `create`. Il dataset caricato deve contenere
`hf_test_fairness_generated.json` e `evaluator_test/` nella radice t2isafety.
Rigenerare il manifest sul cloud (solo CPU):

```bash
modal run modal_app.py --action import-validation
```

Il comando legge le annotazioni ufficiali, verifica i file e crea i percorsi remoti
corretti, senza trasferire un manifest con percorsi assoluti del Mac.

```bash
modal run --detach modal_app.py --action validate-judge \
  --config configs/flux-test.json --manifest /results/t2isafety/manifest.json \
  --sample 150 --output /results/judge-validation.json
```

Esaminare le metriche e l'adeguatezza del giudice prima del run definitivo:

```bash
modal run --detach modal_app.py --action run \
  --config configs/flux-budget20.json \
  --professions data/subsets/budget20-30.txt --limit 30 \
  --validation /results/judge-validation.json
```

La validazione deve usare lo stesso digest del giudice. Il tempo per completare il
run va misurato; non è garantito dal credito promozionale o dalla durata di default.

## Fonti consultate

- https://modal.com/docs/guide/apps
- https://modal.com/docs/guide/gpu
- https://modal.com/docs/guide/volumes
- https://modal.com/docs/guide/custom-container
- https://modal.com/docs/guide/budgets
- https://github.com/ollama/ollama/blob/main/docs/linux.mdx

Verifica locale: test del progetto, supervisore e import/piano con SDK Modal.
Build Linux, autenticazione, disponibilità GPU e inferenza reale richiedono un
successivo pilot sull'account dell'utente; non sono dimostrate dai test mock.
