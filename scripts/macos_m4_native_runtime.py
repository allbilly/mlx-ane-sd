"""JSON pipe to the standalone Swift full-ANE runner, loaded once per bench."""

from __future__ import annotations
import json
import subprocess
from pathlib import Path


class NativeStack:
    def __init__(self, directory, tokenizer):
        self.directory = Path(directory)
        self.tokenizer = tokenizer
        self.manifest = json.loads((self.directory / "manifest.json").read_text())
        self.process = subprocess.Popen(
            [str(self.directory / "m1-full-ane"), str(self.directory)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
        )
        self.ready = self.read()
        if not self.ready.get("ready"):
            raise RuntimeError(f"Native runner not ready: {self.ready}")

    def read(self):
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"Swift runner exited: {self.process.poll()}")
        result = json.loads(line)
        if "error" in result:
            raise RuntimeError(result["error"])
        return result

    def generate(self, prompt, limit, speculative):
        tok = self.tokenizer
        request = {
            "prompt_ids": tok.encode(prompt),
            "limit": limit,
            "speculative": speculative,
            "eos": sorted(tok.eos_token_ids),
        }
        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        result = self.read()
        result["text"] = tok.decode(result["tokens"])
        return result

    def close(self):
        self.process.stdin.close()
        try:
            self.process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            self.process.terminate()
            self.process.wait(timeout=10)
