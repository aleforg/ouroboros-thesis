"""Supervisione del processo remoto; importabile e verificabile senza Modal."""

import subprocess
import time
import urllib.error
import urllib.request


def checked_process(command, *, deadline, commit, interval=30):
    """Conserva output e termina il figlio prima del timeout del container."""
    process = subprocess.Popen(command)
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Limite di durata raggiunto; consultare il checkpoint prima della ripresa")
            try:
                status = process.wait(timeout=min(interval, remaining))
                if status:
                    raise subprocess.CalledProcessError(status, command)
                return
            except subprocess.TimeoutExpired:
                commit()
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
        commit()


def wait_for_ollama(process, timeout=60):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Ollama è terminato durante l'avvio; consultare i log")
        try:
            with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=2):
                return
        except (urllib.error.URLError, TimeoutError):
            time.sleep(1)
    raise TimeoutError("Ollama non è pronto dopo 60 secondi")
