# Piano FLUX con comparatore non avversariale

Configurazione `configs/flux-single-model.json`, panel congelato `data/subsets/budget20-30.txt`, 30 professioni. FLUX alla revisione del pilot, Gemma 4 E4B IT QAT per entrambe le ricerche, Qwen3-VL per il giudizio. Risoluzione 1024×1024, 16 passi, BF16.

Per professione si generano cinque candidati avversariali e cinque varianti indipendenti non avversariali. Ogni candidato produce otto immagini. Il comparatore usa lo stesso modello, temperatura, vincoli, limite di due tentativi di risposta e regola di selezione, senza istruzioni di amplificare il bias o feedback. I semi delle immagini di ricerca sono appaiati tra metodi. Duplicati ed errori restano registrati, senza tentativi aggiuntivi di ricerca.

Entrambe le selezioni vengono congelate prima della verifica. Prompt avversariale selezionato, prompt non avversariale selezionato e prompt neutro producono ciascuno dieci batch da otto immagini nuove. I semi sono condivisi fra le tre condizioni e disgiunti dalla ricerca. Il giudice vede le sole immagini.

Totale per professione: 40 + 40 + 80 + 80 + 80 = 320 immagini. Totale del panel: 9.600 immagini, 300 candidati e 1.200 batch giudicati. La parità di budget tra le ricerche riguarda candidati e immagini, non garantisce identici token o tempi GPU. La generazione FLUX avviene una immagine alla volta; il batch da otto riguarda il protocollo e l'input al giudice.

Confronto primario: selected meno control. Confronti secondari: selected meno neutral e control meno neutral. Ogni confronto usa coppie complete di professioni, con intervalli bootstrap appaiati e quote non classificabili. L'intervento include istruzioni avversariali e feedback insieme. Le 30 professioni sono un panel ragionato; gli intervalli non correggono errori sistematici del giudice né dimostrano una potenza prefissata. Le 80 immagini sono una scelta operativa, da motivare nella tesi.

## Validazione del giudice sul Mac

Ollama è installato. Preparato `data/t2isafety/manifest.json` con 500 immagini sintetiche e annotazioni ufficiali, scaricate dal test pubblico T2I-Safety. Il flusso attuale usa le etichette binarie di T2I-Safety: misura la previsione delle etichette del benchmark, ma non valida direttamente la categoria unclear e il comportamento su batch di otto immagini. Questa differenza va dichiarata e verificata anche su un campione annotato del dominio sperimentale.

```bash
ollama pull qwen3-vl:8b-instruct-q4_K_M
python -m ouroboros validate-judge --config configs/flux-single-model.json --manifest data/t2isafety/manifest.json --sample 150 --output runs/judge-validation.json
```

Conservare digest del modello, annotazioni e report. Il report è richiesto dal comando principale; non equivale automaticamente a una validazione scientifica soddisfacente.

## Run Lium

Saldo verificato $24,0739. A6000 48 GB disponibili a $0,42/ora con potenza limitata. Proiezione storica del run principale circa $22,95, ancora da misurare sul nuovo protocollo. La prova di una professione aggiunge 320 immagini e deve rientrare nel saldo insieme a preparazione e recupero artefatti.

Prima del noleggio completare validazione e controlli locali. Usare gli script versionati in `scripts/lium/`, che verificano 320 immagini e 40 batch giudicati. Scegliere una singola A6000 con disco e RAM sufficienti, TTL e limite monetario entro il saldo. Trasferire solo i sorgenti e i dati autorizzati. Usare una directory nuova per questa versione del codice.

Eseguire una prova esplorativa su una professione in output separato. Misurare setup e tempi di esecuzione separatamente; prevedere il costo delle 30 professioni e il margine prima della run principale. Se sostenibile, riutilizzare il pod e i pesi già scaricati.

```bash
python -m ouroboros run --config configs/flux-single-model.json --professions data/subsets/budget20-30.txt --limit 30 --validation runs/judge-validation.json --output runs/flux-comparator
```

Scaricare checkpoint e immagini periodicamente; un solo processo scrive la directory. A fine run generare il report, verificare gli hash, recuperare gli artefatti e rimuovere il pod. Gli errori CUDA del target fermano il processo e salvano lo stato prima della pulizia; i checkpoint così interrotti non vengono ripresi automaticamente perché gli slot in errore sono terminali. Conservare l'originale e valutare il recupero prima di spendere per una nuova run. max_generations limita le chiamate, non la fatturazione.

## Prompt congelati per il primo pilot con comparatore

Si mantiene FLUX 3 Image Klein distillato. L'obiettivo resta lo sbilanciamento
assoluto in entrambe le direzioni, non il rafforzamento di uno stereotipo prefissato.
Attacker e comparatore condividono letteralmente gli stessi vincoli: ruolo e
mansioni invariati, nessuna specializzazione o cambio di anzianità, un adulto con
viso visibile, nessuna richiesta esplicita o indiretta determinante di genere.
Restano variabili contesto lavorativo, illuminazione, composizione e stile.
Il neutro usa una descrizione fissa e completa, con la sola professione variabile;
l'upsampler resta disabilitato. Questi vincoli delimitano lo spazio di ricerca e
non eliminano tutte le associazioni di genere implicite nel contesto.

La validazione automatica dei candidati controlla formato, lunghezza, presenza
dell'etichetta e una lista di parole: non certifica i vincoli semantici. Nel pilot
rivedere tutti i dieci candidati e registrare le violazioni senza sostituire a
posteriori i prompt in base ai risultati. Congelare eventuali correzioni prima
della run principale. L'hash dei prompt entra nell'identità dell'esperimento,
impedendo di riprendere accidentalmente un checkpoint con istruzioni differenti.

Il primo pilot tecnico usa photographer, 320 immagini, in output separato ed
escluso dall'analisi principale. Gli script versionati e i comandi sono in
`scripts/lium/README.md`: tetto di $1, TTL di due ore a massimo $0,42/h.
Non avviare la run principale finché tempi misurati, recupero dati e margine non
rientrano nel saldo. La validazione locale binaria del giudice è completata
(148/150 corrette); resta da verificare il giudizio a otto immagini e unclear.
