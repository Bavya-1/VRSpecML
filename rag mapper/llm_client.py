"""Qwen 2.5 7B client for a local inference server.

Backends (auto-detected from the host URL, as in ../llm mapper):
  • host ending in /v1  → OpenAI-compatible API (llama.cpp llama-server, default)
  • any other host      → native Ollama API (/api/chat)

Every call requests JSON constrained by a JSON Schema (grammar-based decoding on
llama.cpp / Ollama), so enum-typed fields such as evidence ids cannot be invented.
"""
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests

import config

logger = logging.getLogger(__name__)
_HEADERS = {"Content-Type": "application/json", "User-Agent": "VRSpecML-RAG"}


class LLMError(RuntimeError):
    pass


def is_openai_compatible(host: str) -> bool:
    return host.rstrip("/").lower().endswith("/v1")


def check_connection(host: str, timeout: float = 5) -> Dict[str, Any]:
    """Ping the server: {"ok", "backend", "models", "error"}."""
    host = host.rstrip("/")
    openai = is_openai_compatible(host)
    url = f"{host}/models" if openai else f"{host}/api/tags"
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=timeout)
    except requests.exceptions.RequestException:
        return {"ok": False, "backend": "llama-server" if openai else "ollama", "models": [],
                "error": f"No Qwen server reachable at {host}"}
    if resp.status_code == 503:
        return {"ok": False, "backend": "llama-server", "models": [], "error": "Model is still loading"}
    if resp.status_code != 200:
        return {"ok": False, "backend": "", "models": [], "error": f"HTTP {resp.status_code}: {resp.text[:160]}"}
    data = resp.json()
    models = ([m.get("id", "") for m in data.get("data", [])] if openai
              else [m.get("name", "") for m in data.get("models", [])])
    return {"ok": True, "backend": "llama-server" if openai else "ollama", "models": models, "error": None}


def resolve_model(host: str, preferred: str) -> str:
    """Pick the Qwen model id actually served by the host."""
    info = check_connection(host)
    models = info.get("models") or []
    if preferred in models:
        return preferred
    qwen7 = [m for m in models if "qwen" in m.lower() and "7b" in m.lower()]
    if qwen7:
        return qwen7[0]
    if is_openai_compatible(host) and models:
        return models[0]  # llama-server serves exactly one model
    return preferred if is_openai_compatible(host) else config.OLLAMA_QWEN_MODEL


def clean_json_text(text: str) -> str:
    """Strip <think> blocks, code fences and prose around the outermost JSON object."""
    cleaned = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S).strip()
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned).strip()
    first = cleaned.find("{")
    if first > 0:
        cleaned = cleaned[first:]
    return cleaned


def repair_truncated_json(text: str) -> Optional[Any]:
    """Recover a JSON object cut off by the token limit (keeps complete items only)."""
    text = clean_json_text(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Walk back over the ends of complete objects, then close the open brackets.
    cuts = [i + 1 for i, ch in enumerate(text) if ch == "}"][::-1][:400]
    for cut in cuts:
        head = text[:cut].rstrip().rstrip(",")
        stack, in_str, esc = [], False, False
        for ch in head:
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
            elif ch == '"':
                in_str = True
            elif ch in "{[":
                stack.append(ch)
            elif ch in "}]":
                if stack:
                    stack.pop()
        if in_str:
            continue
        closing = "".join("}" if c == "{" else "]" for c in reversed(stack))
        try:
            return json.loads(head + closing)
        except json.JSONDecodeError:
            continue
    return None


class QwenClient:
    def __init__(self, host: str = config.DEFAULT_LLM_HOST, model: str = config.DEFAULT_LLM_MODEL,
                 temperature: float = config.LLM_TEMPERATURE, seed: int = config.LLM_SEED,
                 timeout: int = config.LLM_TIMEOUT):
        self.host = host.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.seed = seed
        self.timeout = timeout
        self.openai = is_openai_compatible(self.host)
        self._stats_lock = threading.Lock()
        self.stats: List[Dict[str, Any]] = []

    def _post(self, messages: List[Dict[str, str]], schema: Optional[Dict[str, Any]], max_tokens: int):
        if self.openai:
            body: Dict[str, Any] = {
                "model": self.model, "messages": messages, "temperature": self.temperature,
                "seed": self.seed, "max_tokens": max_tokens,
                "response_format": ({"type": "json_schema", "json_schema": {"name": "output", "schema": schema, "strict": True}}
                                    if schema else {"type": "json_object"}),
            }
            url = f"{self.host}/chat/completions"
        else:
            body = {
                "model": self.model, "messages": messages, "stream": False,
                "format": schema if schema else "json",
                "options": {"temperature": self.temperature, "seed": self.seed,
                            "num_predict": max_tokens, "num_ctx": 16384},
            }
            url = f"{self.host}/api/chat"
        try:
            resp = requests.post(url, headers=_HEADERS, json=body, timeout=self.timeout)
        except requests.exceptions.ConnectionError as exc:
            raise LLMError(f"Cannot reach the Qwen server at {self.host}. Start it from the sidebar or run ./run.sh.") from exc
        except requests.exceptions.Timeout as exc:
            raise LLMError(f"Qwen request timed out after {self.timeout}s") from exc
        if resp.status_code == 404 and not self.openai:
            raise LLMError(f"Model '{self.model}' is not pulled in Ollama. Run: ollama pull {config.OLLAMA_QWEN_MODEL}")
        if resp.status_code != 200:
            raise LLMError(f"Qwen server error HTTP {resp.status_code}: {resp.text[:300]}")
        data = resp.json()
        if self.openai:
            choice = (data.get("choices") or [{}])[0]
            content = choice.get("message", {}).get("content", "") or ""
            truncated = choice.get("finish_reason") == "length"
            usage = data.get("usage", {})
            prompt_toks, out_toks = usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)
        else:
            content = data.get("message", {}).get("content", "") or ""
            truncated = data.get("done_reason") == "length"
            prompt_toks, out_toks = data.get("prompt_eval_count", 0), data.get("eval_count", 0)
        return content, truncated, prompt_toks, out_toks

    def chat_json(self, system: str, user: str, schema: Optional[Dict[str, Any]] = None,
                  max_tokens: int = 2048, task: str = "") -> Dict[str, Any]:
        """One schema-constrained chat completion, parsed to a dict (one retry on failure)."""
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        last_error = ""
        for attempt in range(2):
            t0 = time.time()
            content, truncated, p_toks, o_toks = self._post(messages, schema, max_tokens)
            with self._stats_lock:
                self.stats.append({"task": task, "prompt_tokens": p_toks, "completion_tokens": o_toks,
                                   "seconds": round(time.time() - t0, 2), "truncated": truncated})
            parsed = repair_truncated_json(content)
            if isinstance(parsed, dict):
                if truncated:
                    logger.warning("%s: output hit max_tokens; kept complete items only", task)
                return parsed
            last_error = f"unparseable output: {content[:200]!r}"
            logger.warning("%s attempt %d: %s", task, attempt + 1, last_error)
            max_tokens = int(max_tokens * 1.5)
        raise LLMError(f"{task or 'LLM call'} failed: {last_error}")

    def usage_summary(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = list(self.stats)
        return {
            "calls": len(stats),
            "prompt_tokens": sum(s["prompt_tokens"] for s in stats),
            "completion_tokens": sum(s["completion_tokens"] for s in stats),
            "seconds": round(sum(s["seconds"] for s in stats), 1),
        }
