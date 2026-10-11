# FLASH-MOE

**Prototipo per eseguire un modello linguistico MoE in locale su hardware
Intel, usando OpenVINO.**

Il progetto sperimenta come distribuire il lavoro tra processore e grafica
integrata, caricando dal disco solo una parte dei pesi del modello alla volta.
L'obiettivo e' capire fin dove si puo' arrivare con un PC comune, senza
richiedere una scheda grafica dedicata con molta memoria.

> **Stato: in sviluppo.** Il programma genera gia' risposte a prompt reali,
> ma velocita', qualita' delle risposte e stabilita' vanno ancora verificate
> meglio. Aspettati tempi lunghi, cambiamenti e possibili problemi.

## In breve

- Legge un modello in formato GGUF e permette di fargli domande da PowerShell.
- Usa OpenVINO per eseguire i calcoli su CPU o su un dispositivo Intel
  supportato.
- Il modello e' di tipo **MoE** (*Mixture of Experts*): a ogni passo usa solo
  alcuni dei suoi gruppi di pesi, chiamati *expert*, invece di attivarli tutti.
- L'interfaccia mostra separatamente il tempo necessario per leggere il
  prompt e quello impiegato per generare la risposta.

## Stato e risultati

Il primo modello provato e' **Hermes 3.6 35B A3B**, una variante Qwen3.5-MoE
in formato GGUF. Il file del modello non e' incluso nel repository: e' molto
grande e va ottenuto separatamente.

La prima prova manuale su un **Intel Core Ultra 5 226V con grafica integrata
Intel Arc 130V** ha generato risposte comprensibili. Nei prompt provati la
velocita' e' variata da circa **0,2 a 0,8 token al secondo**. Sono misure
indicative, non un benchmark controllato: cambiano con la domanda, la
lunghezza della risposta e lo stato della cache.

Le prove automatiche controllano separatamente componenti come attenzione,
dequantizzazione e percorso DeltaNet. Non equivalgono a una validazione
completa della qualita' delle risposte o dell'affidabilita' su ogni PC.

## Requisiti

- Windows 10 o 11 a 64 bit.
- Python 3.14 e i pacchetti elencati in `requirements.txt`. Python 3.14.7 e'
  la versione usata nelle prove; altre versioni potrebbero funzionare, ma non
  sono state verificate con la stessa configurazione.
- Un file GGUF compatibile. Il modello usato nelle prove e' indicato sopra.
- Driver Intel aggiornati se vuoi usare la grafica integrata.

Per installare i pacchetti da PowerShell, nella cartella del progetto:

```powershell
py -3.14 -m pip install -r requirements.txt
```

Se il comando `py` non e' disponibile, usa `python` al suo posto.

## Preparare il modello

Se il file GGUF si trova nella cartella del progetto, il programma prova a
trovarlo automaticamente. Altrimenti indica il suo percorso prima di avviare
la chat:

```powershell
$env:FLASH_MOE_GGUF = "D:\modelli\modello.gguf"
```

Il progetto **non distribuisce il modello**. Scaricalo da una fonte
autorizzata e controlla le condizioni d'uso previste dal suo autore: la
licenza del modello e quella del codice sono separate.

### Fonte e attribuzione del modello provato

La corrispondenza pubblica piu' vicina al nome del file usato nei test e'
[`LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-GGUF`](https://huggingface.co/LuffyTheFox/Qwen3.6-35B-A3B-Uncensored-Genesis-Hermes-GGUF).
La scheda descrive un modello basato su
[`HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive`](https://huggingface.co/HauhauCS/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive),
con un contributo Hermes attribuito a
[`DJLougen`](https://huggingface.co/DJLougen/hermes-qwen3.5-35b-a3b-GGUF).
Le schede di LuffyTheFox e della base HauhauCS dichiarano Apache-2.0 per i
rispettivi repository.

Il file presente nei test si chiama
`Hermes3.6-35B-A3B-Uncensored-Genesis-Final-APEX-Compact.gguf`, mentre nel
repository LuffyTheFox la variante corrispondente e' elencata senza `Final`
nel nome. La corrispondenza e' quindi probabile, ma non conferma da sola che
si tratti della stessa revisione o degli stessi pesi. Il modello e' escluso
dal repository del progetto; per usarlo o ridistribuirlo, verifica sempre la
fonte e i termini associati al file effettivamente scaricato.

## Avviare una conversazione

```powershell
py -3.14 chat.py --device GPU --ssm-device GPU --attention-device GPU --cache 320 --weight-cache 384 --max-tokens 256 --temp 0
```

La chat chiede un prompt dopo il simbolo `>`. Per uscire, scrivi `/esce` o
premi `Ctrl+C`.

Il comando seleziona la GPU Intel anche per alcune operazioni interne. Se il
driver o il dispositivo non sono disponibili, prova a omettere le opzioni
`--ssm-device GPU` e `--attention-device GPU`, oppure usa `--device CPU`.

L'assistente riceve un'istruzione per rispondere in modo conciso. `--max-tokens`
e' un **limite massimo**, non la lunghezza obbligatoria: la generazione
termina prima se il modello emette il segnale di fine risposta. Se invece
raggiunge il limite, la risposta puo' restare incompleta; in quel caso
aumentalo, per esempio con `--max-tokens 384`.

Al termine di ogni risposta vengono mostrati i token generati al secondo, il
tempo di lettura del prompt e alcune statistiche della cache. Il caricamento
iniziale puo' richiedere parecchio tempo.

### Opzioni utili

- `--device CPU`, `GPU` o `NPU`: sceglie il dispositivo principale di
  OpenVINO. La disponibilita' dipende da hardware e driver.
- `--max-tokens 256`: stabilisce il massimo di token generabili per risposta.
- `--cache 320`: limita quanti gruppi di pesi quantizzati tenere in memoria.
- `--weight-cache 384`: limita quanti gruppi di pesi gia' convertiti tenere
  in memoria. Ridurre il valore puo' aiutare se il PC e' a corto di RAM.
- `--temp 0`: usa una scelta deterministica dei token; valori maggiori
  introducono piu' variabilita'.
- `--max-seq 512`: imposta la lunghezza massima gestita per il contesto.

## Accelerazione della conversione dei pesi

Su Windows e' possibile compilare il modulo C++ usato per convertire alcuni
formati di pesi:

```powershell
scripts\compile_dequant_k.bat
```

Serve Visual Studio Build Tools con il compilatore C++ x64. Il file generato
non e' incluso nel repository. Se non e' presente, il programma usa il
percorso alternativo basato sui pacchetti Python: puo' essere piu' lento, ma
non richiede la DLL.

## Eseguire alcune verifiche

I test sono script Python; dalla cartella principale si possono lanciare, ad
esempio, cosi':

```powershell
py -3.14 tools\test_attention.py
py -3.14 tools\test_gated_deltanet_real.py
```

Alcuni test richiedono il modello GGUF configurato e possono durare a lungo.
Il secondo test usa il modello configurato; esegui prima i passaggi della
sezione precedente. Queste verifiche controllano singoli componenti e non
sostituiscono una prova completa della chat.

Gli script di confronto con i dump di `llama.cpp` cercano per impostazione
predefinita `lcpp_eval.txt` nella cartella del progetto. Per usare un dump
altrove, imposta `FLASH_MOE_LCPP_EVAL` al suo percorso prima di avviare lo
script.

## Limiti attuali

- Per ora e' stato provato soprattutto un singolo PC Intel; altri hardware
  possono avere prestazioni o compatibilita' diverse.
- La generazione e' lenta rispetto ai servizi online e ai sistemi con GPU
  dedicata.
- Le prestazioni dipendono molto dall'accesso al file del modello e dalla
  memoria disponibile.
- Le istruzioni di risposta concisa aiutano a limitare la lunghezza, ma il
  comportamento resta deciso dal modello; il tetto di token puo' comunque
  interrompere una risposta lunga.
- Non e' ancora un'applicazione pronta per un uso produttivo.

## Licenza del codice

Il codice di questo repository e' distribuito con licenza
[Apache-2.0](./LICENSE). Questa licenza riguarda il codice del progetto, non
i pesi o i dati dei modelli esterni.
