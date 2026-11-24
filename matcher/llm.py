"""LLM behind a small interface (decision 8): complete(prompt) -> (text, usage).

BedrockLLM uses the Converse API. CachedLLM wraps any LLM and stores every reply on disk, keyed by
the SHA-256 of the prompt, so a re-run (or a crash half-way through ~80 calls) never pays twice.
"""
import hashlib, json, os, threading

import boto3
from botocore.config import Config


class BedrockLLM:
    def __init__(self, model_id, profile, region, max_tokens, temperature):
        session = boto3.Session(profile_name=os.environ.get("AWS_PROFILE", profile), region_name=region)
        # long read timeout: a full window of steps can take a minute or more to generate
        self.client = session.client("bedrock-runtime", config=Config(
            read_timeout=300, retries={"max_attempts": 5, "mode": "adaptive"}))
        self.model_id, self.max_tokens, self.temperature = model_id, max_tokens, temperature

    def complete(self, prompt):
        resp = self.client.converse(
            modelId=self.model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.max_tokens, "temperature": self.temperature})
        text = "".join(c.get("text", "") for c in resp["output"]["message"]["content"])
        usage = {"input_tokens": resp["usage"]["inputTokens"], "output_tokens": resp["usage"]["outputTokens"],
                 "stop_reason": resp.get("stopReason")}
        return text, usage


class CachedLLM:
    """Reads a stored reply when the exact same prompt was sent before; otherwise calls the LLM.
    Only replies the caller accepts are stored (see `store`), so an invalid reply is asked again."""

    def __init__(self, llm, cache_dir, model_id):
        self.llm, self.dir, self.model_id = llm, cache_dir, model_id
        self.dir.mkdir(parents=True, exist_ok=True)
        self.hits = self.calls = 0
        self.lock = threading.Lock()  # spans are segmented in parallel threads

    def _path(self, prompt):
        return self.dir / f"{hashlib.sha256((self.model_id + prompt).encode()).hexdigest()}.json"

    def complete(self, prompt):
        path = self._path(prompt)
        if path.exists():
            with self.lock:
                self.hits += 1
            entry = json.loads(path.read_text())
            return entry["reply"], {**entry["usage"], "cached": True}
        with self.lock:
            self.calls += 1
        text, usage = self.llm.complete(prompt)
        return text, {**usage, "cached": False}

    def store(self, prompt, reply, usage):
        entry = {"model_id": self.model_id, "usage": {k: v for k, v in usage.items() if k != "cached"},
                 "prompt": prompt, "reply": reply}
        self._path(prompt).write_text(json.dumps(entry, indent=1))
