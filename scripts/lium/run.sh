#!/usr/bin/env bash
set -euo pipefail
export PYTHONUNBUFFERED=1
export HF_HOME=/workspace/hf
phase=${1:?Usare pilot oppure main}
case "$phase" in
  pilot) panel=data/subsets/lium-pilot.txt; count=1; gate=(--exploratory) ;;
  main) panel=data/subsets/budget20-30.txt; count=30; gate=(--validation /workspace/judge-validation.json) ;;
  *) exit 2 ;;
esac
export OLLAMA_HOST=127.0.0.1:11434
export OLLAMA_MODELS=/workspace/models/ollama
cd /workspace/ouroboros
mkdir -p /workspace/artifacts
ollama serve > /workspace/artifacts/ollama.log 2>&1 &
server_pid=$!
trap 'kill "$server_pid" 2>/dev/null || true' EXIT
python - <<'CHECK'
import time, urllib.request
for attempt in range(60):
    try:
        urllib.request.urlopen('http://127.0.0.1:11434/api/tags',timeout=2).close()
        break
    except OSError:
        time.sleep(1)
else:
    raise RuntimeError('Ollama non pronto')
CHECK
ollama pull gemma4:e4b-it-qat > /workspace/artifacts/pull-attacker.log 2>&1
ollama pull qwen3-vl:8b-instruct-q4_K_M > /workspace/artifacts/pull-judge.log 2>&1
ollama list > /workspace/artifacts/models.txt
nvidia-smi --query-gpu=timestamp,memory.used,memory.total,utilization.gpu,power.draw --format=csv -l 5 > "/workspace/artifacts/$phase-gpu-monitor.csv" &
monitor_pid=$!
trap 'kill "$monitor_pid" "$server_pid" 2>/dev/null || true' EXIT

output="/workspace/artifacts/$phase"
if [[ -d "$output" && "$phase" == pilot ]]; then
    echo 'Pilot già avviato: conservare gli artefatti prima di un nuovo tentativo.' >&2
    exit 1
fi
if [[ "$phase" == pilot ]]; then
    /usr/bin/time -o /workspace/artifacts/flux-preflight.time -v python - <<'PRELOAD'
from ouroboros.config import load_config
from ouroboros.models import DiffusersTarget, model_provenance

config = load_config('configs/flux-single-model.json')
model_provenance(config)
target = DiffusersTarget(config)
target.load()
target.unload()
PRELOAD
fi
/usr/bin/time -o "/workspace/artifacts/$phase.time" -v python -m ouroboros run --config configs/flux-single-model.json \
    --professions "$panel" --limit "$count" "${gate[@]}" --output "$output"
python - "$output" "$count" <<'CHECK'
from pathlib import Path
import sys
from ouroboros.analysis import make_report
from ouroboros.storage import read_json, verify_images

checkpoint, = Path(sys.argv[1]).glob('*/checkpoint.json')
state = read_json(checkpoint)
verify_images(checkpoint.parent, state)
report = make_report(checkpoint.parent)
count = int(sys.argv[2])
images = sum(len(record['images']) for record in state['records'])
print(f"Immagini verificate: {images}; coppie primarie: "
      f"{report['comparisons']['selected_minus_control']['n_pairs']}/{count}")
if not state['finished'] or len(state['completed']) != count:
    raise SystemExit("Run incompleto; recuperare checkpoint e log")
if count == 1 and (images != 320 or report['statuses'] != {'judged': 40}
                   or report['comparisons']['selected_minus_control']['n_pairs'] != 1):
    raise SystemExit("Pilot incompleto; controllare report e log")
CHECK
