"""Server CUDA persistente per fase; solo pesi locali già quantizzati."""

import base64
import io
import json
from pathlib import Path
import socket
import subprocess
import tempfile
import time
import urllib.error
import urllib.request

from .models import request_json
from .storage import sha256


def artifact_provenance(config):
    return {name: {"path": str(Path(getattr(config, name)).resolve()),
                   "sha256": sha256(getattr(config, name))}
            for name in ("sd_server", "diffusion_weights", "encoder_weights", "vae_weights")}


class SDCppTarget:
    def __init__(self, config):
        self.config = config
        self.process = None
        self.log = None
        self.url = None
        self.timings = {"load_seconds": 0.0, "generation_seconds": 0.0, "unload_seconds": 0.0,
                        "loads": 0, "images": 0}

    def command(self, port):
        c = self.config
        command = [str(Path(c.sd_server).resolve()),
                   "--diffusion-model", str(Path(c.diffusion_weights).resolve()),
                   "--llm", str(Path(c.encoder_weights).resolve()),
                   "--vae", str(Path(c.vae_weights).resolve()),
                   "--listen-ip", "127.0.0.1", "--listen-port", str(port),
                   "--rng", "cpu", "--flash-attn", "on", "--vae-tiling"]
        if c.cpu_offload:
            command.append("--offload-to-cpu")
        if c.target == "flux":
            command.extend(["--flow-shift", "6.93"])
        if c.qwen_turbo:
            command.extend(["--sigmas", "qwen-image-2.1-turbo"])
        return command

    def load(self):
        if self.process is not None:
            if self.process.poll() is not None:
                raise RuntimeError(f"sd-server terminato; consultare {self.log.name}")
            return
        started = time.perf_counter()
        for name in ("sd_server", "diffusion_weights", "encoder_weights", "vae_weights"):
            if not Path(getattr(self.config, name)).is_file():
                raise FileNotFoundError(f"{name}: {getattr(self.config, name)}")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        self.url = f"http://127.0.0.1:{port}"
        self.log = tempfile.NamedTemporaryFile(prefix="ouroboros-sdcpp-", suffix=".log", delete=False)
        try:
            self.process = subprocess.Popen(self.command(port), stdout=self.log, stderr=subprocess.STDOUT)
            deadline = time.monotonic() + 600
            while time.monotonic() < deadline:
                if self.process.poll() is not None:
                    raise RuntimeError(f"Avvio sd-server fallito; consultare {self.log.name}")
                try:
                    # Endpoint senza inferenza: il server è pronto dopo il caricamento.
                    with urllib.request.urlopen(
                            self.url + "/sdapi/v1/options", timeout=2) as response:
                        if response.status == 200:
                            self.timings["load_seconds"] += time.perf_counter() - started
                            self.timings["loads"] += 1
                            return
                except (urllib.error.URLError, TimeoutError):
                    pass
                time.sleep(.2)
            raise TimeoutError(f"Caricamento sd-server oltre 600 secondi; consultare {self.log.name}")
        except BaseException:
            self.unload()
            raise

    def generate(self, prompt, seed):
        from PIL import Image

        # Il server interpreta questo tag come parametri: un prompt non può cambiarli.
        if "sd_cpp_extra_args" in prompt:
            raise ValueError("Il prompt contiene un override dei parametri sd.cpp")
        if type(seed) is not int or not 0 <= seed < 2**32:
            raise ValueError("Seed fuori dallo spazio del protocollo")
        self.load()
        c = self.config
        started = time.perf_counter()
        # sdapi legge il seed come int32; il parser nativo conserva int64 e l'holdout.
        native_seed = "<sd_cpp_extra_args>" + json.dumps({"seed": seed}) + "</sd_cpp_extra_args>"
        result = request_json(self.url + "/sdapi/v1/txt2img", {
            "prompt": prompt + native_seed, "negative_prompt": "", "width": c.width, "height": c.height,
            "steps": c.steps, "cfg_scale": 1.0 if c.target == "flux" or c.qwen_turbo else 6.0,
            "seed": 0, "batch_size": 1, "sampler_name": "Euler"})
        info = json.loads(result.get("info", "{}"))
        if info.get("all_seeds") != [seed]:
            raise RuntimeError("sd-server non conferma il seed richiesto: verificare la versione")
        images = result.get("images", [])
        if len(images) != 1:
            raise RuntimeError("sd-server deve restituire esattamente una immagine")
        with Image.open(io.BytesIO(base64.b64decode(images[0], validate=True))) as image:
            if image.size != (c.width, c.height):
                raise RuntimeError(f"Dimensioni inattese da sd-server: {image.size}")
            self.timings["generation_seconds"] += time.perf_counter() - started
            self.timings["images"] += 1
            return image.copy()

    def unload(self):
        started = time.perf_counter()
        try:
            if self.process is not None:
                if self.process.poll() is None:
                    self.process.terminate()
                    try:
                        self.process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        self.process.kill()
                        self.process.wait()
                self.process = None
        finally:
            if self.log is not None:
                self.log.close()
            self.timings["unload_seconds"] += time.perf_counter() - started
