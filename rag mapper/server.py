"""Start / stop / probe the local Qwen 2.5 7B llama-server (llama.cpp, GPU offload).

CLI:  python server.py start [--wait] | stop | status
"""
import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from typing import Any, Dict, Optional

import config
from llm_client import check_connection

PID_FILE = os.path.join(config.RUN_DIR, "llama-server.pid")
LOG_FILE = os.path.join(config.RUN_DIR, "llama-server.log")
LOCAL_HOST = f"http://127.0.0.1:{config.LLAMA_SERVER_PORT}/v1"


def find_llama_server() -> Optional[str]:
    candidates = [shutil.which("llama-server"), os.path.join(os.path.dirname(sys.executable), "llama-server")]
    for c in candidates:
        if c and os.path.isfile(c) and os.access(c, os.X_OK):
            return c
    return None


def _read_pid() -> Optional[int]:
    try:
        with open(PID_FILE) as fh:
            pid = int(fh.read().strip())
        os.kill(pid, 0)
        return pid
    except (OSError, ValueError):
        return None


def status(host: str = LOCAL_HOST) -> Dict[str, Any]:
    info = check_connection(host)
    info["pid"] = _read_pid()
    return info


def start(wait: bool = False, timeout: int = 240) -> Dict[str, Any]:
    """Launch llama-server with the Qwen GGUF unless a server already answers."""
    current = status()
    if current["ok"]:
        return {"started": False, "message": "Qwen server already running", **current}
    if _read_pid():
        if wait:
            return wait_ready(timeout)
        return {"started": False, "ok": False, "message": "Qwen server is loading"}
    binary = find_llama_server()
    if not binary:
        return {"started": False, "ok": False, "message": "llama-server not found (conda install -c conda-forge llama.cpp)"}
    if not os.path.isfile(config.QWEN_GGUF):
        return {"started": False, "ok": False, "message": f"Qwen GGUF not found: {config.QWEN_GGUF}"}
    os.makedirs(config.RUN_DIR, exist_ok=True)
    cmd = [
        binary, "-m", config.QWEN_GGUF, "--host", "127.0.0.1", "--port", str(config.LLAMA_SERVER_PORT),
        "-ngl", "99", "-c", str(config.LLAMA_SERVER_CTX), "-np", str(config.LLAMA_SERVER_PARALLEL),
        "--alias", config.DEFAULT_LLM_MODEL,
    ]
    log = open(LOG_FILE, "ab")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    with open(PID_FILE, "w") as fh:
        fh.write(str(proc.pid))
    if wait:
        return wait_ready(timeout)
    return {"started": True, "ok": False, "message": "Qwen server starting", "pid": proc.pid}


def wait_ready(timeout: int = 240) -> Dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        info = status()
        if info["ok"]:
            return {"started": True, "message": "Qwen server ready", **info}
        if not _read_pid():
            return {"started": False, "ok": False, "message": f"llama-server exited — see {LOG_FILE}"}
        time.sleep(2)
    return {"started": True, "ok": False, "message": "Timed out waiting for Qwen server"}


def stop() -> bool:
    pid = _read_pid()
    if not pid:
        return False
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except OSError:
        os.kill(pid, signal.SIGTERM)
    try:
        os.remove(PID_FILE)
    except OSError:
        pass
    return True


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["start", "stop", "status"])
    ap.add_argument("--wait", action="store_true")
    args = ap.parse_args()
    if args.action == "start":
        res = start(wait=args.wait)
        print(res.get("message"))
        sys.exit(0 if res.get("ok") or not args.wait else 1)
    if args.action == "stop":
        print("stopped" if stop() else "no managed server running")
    else:
        s = status()
        print(("online: " + ", ".join(s["models"])) if s["ok"] else f"offline: {s['error']}")
