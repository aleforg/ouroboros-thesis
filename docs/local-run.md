# Esecuzione locale: RTX 3070 8 GB, RAM 16 GB

Questi preset sono candidati per un pilot, non configurazioni certificate sulla GPU.
Usano 1024×1024, generazione di una immagine alla volta, 8 immagini per batch
sperimentale, 5 tentativi e 5 batch di verifica per condizione. Il pilot di due
professioni richiede al massimo 240 immagini per target. Non eseguirlo insieme
ad altre applicazioni GPU. Usare SSD locale; non tenere tutti i modelli in RAM.

## Preparazione

Installare il progetto normalmente (`pip install -e .`), Ollama con i modelli
indicati nei preset, e una build **CUDA** recente di `sd-server` da
[stable-diffusion.cpp](https://github.com/leejet/stable-diffusion.cpp).
Per questo backend non servono PyTorch e Diffusers. Conservare il pacchetto
completo del runtime (incluse eventuali DLL) e la revisione/build scelta.
Non aggiornare runtime, pesi o Ollama durante un esperimento.

Il progetto avvia e arresta il proprio server locale: non avviarlo manualmente.
Il processo resta attivo per tutta una fase e ascolta soltanto su loopback.

Modificare i percorsi nei preset prima del pilot:

| Preset | Generatore | Encoder | Passi |
| --- | --- | --- | --- |
| `configs/flux-local.json` | FLUX 3 Image Klein **5.6B distilled** GGUF Q4 | Qwen3-VL-4B-Instruct GGUF Q4 | 16 |
| `configs/qwen-local.json` | Qwen-Image-2.1 GGUF Q4 | Qwen3-VL-8B GGUF Q4 | 40 |
| `configs/qwen-turbo-local.json` | Qwen-Image-2.1 **Turbo** GGUF Q4 | Qwen3-VL-8B GGUF Q4 | 8 |

I percorsi dei pesi nei JSON sono segnaposto locali: sostituirli con i file
effettivamente scaricati. FLUX 3 usa una directory distinta `models/flux3/`,
il checkpoint distilled (non Base), l'encoder Qwen3-VL-4B-Instruct e
`ae.safetensors` di FLUX 3 (architettura VAE FLUX.2). I vecchi pesi del
generatore e dell'encoder FLUX 2 Klein non sono compatibili.
Serve una build sd.cpp che includa il supporto FLUX 3: aggiornare solo i
pesi non basta. Il backend usa Euler, CFG 1 e flow shift 6.93.
Come nel percorso Diffusers, il prompt viene passato invariato, senza
upsampler, per conservare il protocollo dell'audit (vedi README).
Il passaggio da 4 a 16 passi e al generatore 5.6B richiede un nuovo pilot:
la compatibilità con RTX 3070 8 GB e RAM 16 GB non è ancora misurata.
Su Windows impostare `sd_server` al percorso di `sd-server.exe` (nel JSON usare
slash `/` oppure backslash raddoppiati). Usare il VAE originale corretto per
ciascuna architettura, senza decoder sostitutivi. Per Qwen 2.1 non è valido il
VAE della prima Qwen-Image. Scegliere una quantizzazione precisa e registrarla
insieme alla fonte dei pesi; l'applicazione salva SHA256 dei tre file e del
binario, ma non può dedurre dal nome se il checkpoint sia realmente Turbo.

Guide e pesi collegati dagli autori del runtime:

- [FLUX 3 Image Klein](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/flux3.md)
- [Qwen-Image-2.1 e Turbo](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/qwen_image_2.1.md)
- [Prestazioni, Flash Attention e VAE tiling](https://github.com/leejet/stable-diffusion.cpp/blob/master/docs/performance.md)
- [API server](https://github.com/leejet/stable-diffusion.cpp/blob/master/examples/server/api.md)

Il preset mantiene il giudice esistente `qwen3-vl:8b-instruct-q4_K_M` e
l'attaccante `ministral-3:8b`. Scaricarli con `ollama pull` prima di iniziare.
Il contesto rimane 16384, come nel protocollo originale. Ridurlo alla cieca può
troncare immagini o feedback; `ollama_context` è esplicito e una sua modifica
richiede una validazione coerente. Il giudice vede ancora le otto immagini
insieme: dividere le richieste cambierebbe la modalità di valutazione.

## Controllo e pilot

Dalla radice del progetto, con ambiente Python attivato:

```bash
python -m ouroboros doctor --config configs/flux-local.json
python -m ouroboros run --config configs/flux-local.json --professions data/subsets/budget20-30.txt --limit 2 --output runs/local-pilot --exploratory
```

`doctor` controlla modelli Ollama e file locali e calcola gli hash; non certifica
che la build sia CUDA né che il modello entri in memoria. Il pilot è inferenza
reale. Il limite 240 dei preset copre **due** professioni: specificare `--limit 2`.
Provare FLUX per primo, poi Qwen in una directory di output distinta.

Per 30 professioni impostare `max_generations: 3600`, `mode: "full"`, passare
`--limit 30` e usare `--validation` con la validazione del giudice. Due target
richiedono 7200 immagini. Selezionare prima quale Qwen valutare: standard e Turbo
sono target diversi, non varianti intercambiabili a metà esperimento.

## Memoria e tempi

Il backend usa pesi già quantizzati, `--offload-to-cpu`, Flash Attention e VAE
tiling. Queste scelte riducono la pressione sulla memoria ma non garantiscono
che ogni componente o il giudice a otto immagini entri in 8 GB. L'offload può
ancora trasferire pesi per immagine. Non usiamo un processo nuovo per immagine,
né cache approssimate del denoising, né quantizzazione al caricamento.

Il checkpoint contiene:

- `phase_timings`: durata delle fasi, comprensiva di caricamento e scaricamento;
- `target_timings_current_session`: tempi cumulativi del generatore per
  caricamento, generazioni riuscite e scaricamento, oltre a conteggi.
  Questi ultimi ripartono quando si riavvia Python: non sono il totale storico.

Il log di ogni processo è `ouroboros-sdcpp-*.log` nella directory temporanea del
sistema; in caso di errore di avvio il messaggio indica il percorso. Conservare
anche i log di Ollama. Per GPU e RAM misurare sul laptop con `nvidia-smi` e il
monitor di sistema durante **tutte** le fasi. I tempi del mock non sono benchmark.
Un timeout HTTP è di 600 secondi e l'avvio del server ha un limite di 600 secondi.
Su una configurazione che li supera, il pilot registra errori: non reinterpretarli
come rifiuti del modello né come risultati validi.

## Ordine, ripresa e validità

Con `group_by_phase=true` (richiede `sequential=true`), ogni turno esegue prima
le proposte per tutte le professioni ammesse dal budget, poi le immagini e infine
i giudizi. Ogni professione conserva soltanto il proprio feedback; il turno
successivo parte dopo i giudizi del precedente. Le selezioni sono salvate prima
di qualunque immagine di verifica. La verifica viene generata in un solo blocco,
poi giudicata mantenendo le richieste originali separate per batch.

Per 30 professioni e 5 tentativi: fino a 5 caricamenti del generatore per la
ricerca e uno per la verifica, a fronte dei 450 batch totali. Non è una stima del
fattore di accelerazione: il tempo di inferenza resta da misurare.

Il budget viene prenotato per professioni intere prima di iniziare il gruppo.
Le immagini salvate e i record giudicati non vengono ripetuti dopo Ctrl+C.
Una chiamata interrotta resta conteggiata, quindi lasciare margine al budget
prima del run se sono previste interruzioni; un budget esatto può impedire di
completare il gruppo dopo un'interruzione. Non modificare la configurazione per
riprendere: è parte dell'identità del run. Il codice modificato blocca la ripresa
nella stessa directory, come già previsto dal progetto.

La route sdapi del runtime consultato legge il campo esterno `seed` come int32.
Per conservare l'intervallo di verifica (>= 2^31), l'adapter usa il campo `seed`
del parser nativo tramite `sd_cpp_extra_args` e verifica `info.all_seeds` nella
risposta. Versioni che non confermano il seme falliscono esplicitamente.
I prompt dell'attaccante non possono introdurre quel tag di override.

I preset Diffusers e Modal esistenti mantengono il loro ordine precedente.
Quantizzazione, backend, VAE tiling e variante Turbo vanno dichiarati nella tesi;
lo stesso seme non implica immagini uguali tra Diffusers e sd.cpp. Validare il
giudice effettivo e conservare versioni, configurazioni, hash e output.

## Verifica software

```bash
python -m unittest discover -s tests -q
```

I test confrontano risultati e feedback fra i due ordini con fixture deterministiche,
interruzioni durante generazione e giudizio, budget, lifecycle Ollama/server,
semi di verifica e hash dei pesi. La compatibilità CUDA, i picchi RAM/VRAM e la
qualità dei checkpoint quantizzati richiedono il pilot sulla macchina reale.
