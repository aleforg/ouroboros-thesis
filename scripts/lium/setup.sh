#!/usr/bin/env bash
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
export PYTHONUNBUFFERED=1
mkdir -p /workspace/ouroboros /workspace/artifacts
export HF_HOME=/workspace/hf
exec > >(tee /workspace/artifacts/setup.log) 2>&1
date -u
apt-get update -qq
apt-get install -y -qq git curl zstd libgomp1 time
cd /workspace/ouroboros
tar -xzf /workspace/source.tar.gz
python -m pip install -e . -r requirements-models.txt
python -m unittest discover -s tests -q
python - <<'CHECK'
import torch, diffusers
assert torch.cuda.is_available(), 'CUDA unavailable'
assert hasattr(diffusers, 'Flux3DistilledAutoBlocks')
print('GPU:', torch.cuda.get_device_name(), 'VRAM:', torch.cuda.get_device_properties(0).total_memory)
CHECK
curl -fsSL --retry 3 https://ollama.com/download/ollama-linux-amd64.tar.zst -o /workspace/ollama.tar.zst
tar --zstd -xf /workspace/ollama.tar.zst -C /usr
rm /workspace/ollama.tar.zst
python -m pip freeze > /workspace/artifacts/packages.txt
nvidia-smi > /workspace/artifacts/gpu.txt
ollama --version > /workspace/artifacts/ollama-version.txt 2>&1
printf 'photographer\n' > data/subsets/lium-pilot.txt
touch /workspace/artifacts/setup.complete
