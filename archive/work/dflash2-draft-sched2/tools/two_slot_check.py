#!/usr/bin/env python3
"""Two-slot DFlash2 concurrency check (issue #208 campaign).

Starts llama-server with a 2-slot unified target + the DFlash2 draft, fires two
concurrent greedy requests, and reports per-request text hash / timing plus the
server log's per-slot draft acceptance and the draft context's graph reuse.

Usage: python3 two_slot_check.py [--lib fix|base|asis] [--ctx 32768] [--n 256]
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.request

LLAMA = "/home/stew675/llama.cpp"
BIN = os.path.join(LLAMA, "build-rocm/bin")
SERVER = os.path.join(BIN, "llama-server")
REPO = "/home/stew675/llama-cpp-rdna-boosts"
PROMPT = os.path.join(REPO, "prompts/prose-rdna-boosts.txt")
MODEL = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-UD-Q4_K_XL.gguf"
DRAFT = "/llm/models/Qwen3.8/27B/Q4_K_XL/Qwen3.8-27B-DFlash2-Q4_K_M.gguf"
PORT = int(os.environ.get("PORT", "8039"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lib", default="fix", choices=["fix", "base", "asis"])
    ap.add_argument("--ctx", default="32768")
    ap.add_argument("--n", default="256")
    ap.add_argument("--nmax", default="5")
    ap.add_argument("--gpus", default="0,1")
    ap.add_argument("--outdir", default="/tmp/df2/twoslot")
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)

    if a.lib != "asis":
        src = "/tmp/df2/libllama-%s.so" % a.lib
        shutil.copyfile(src, os.path.join(BIN, "libllama.so.0.6.0"))

    log = os.path.join(a.outdir, "server-%s.log" % a.lib)
    cmd = [
        SERVER, "-m", MODEL, "-md", DRAFT,
        "--spec-type", "draft-dflash", "--spec-draft-n-max", a.nmax,
        "-ngl", "99", "-sm", "tensor", "-mg", "0",
        "-c", a.ctx, "-ub", "2048", "-b", "2048", "-fa", "on",
        "-ctk", "q8_0", "-ctv", "q8_0", "-t", "8",
        "-np", "2", "--kv-unified", "-s", "42", "--temp", "0",
        "--host", "127.0.0.1", "--port", str(PORT), "-lv", "4", "--no-webui",
    ]
    env = os.environ.copy()
    env["HIP_VISIBLE_DEVICES"] = a.gpus
    env["LLAMA_DRAFT_GRAPH_DEBUG"] = "1"
    with open(log, "w") as lf:
        srv = subprocess.Popen(cmd, stdout=lf, stderr=subprocess.STDOUT, env=env)

    base = "http://127.0.0.1:%d" % PORT
    try:
        for _ in range(600):
            try:
                urllib.request.urlopen(base + "/health", timeout=1).read()
                break
            except Exception:
                if srv.poll() is not None:
                    raise SystemExit("server exited during load (see %s)" % log)
                time.sleep(0.5)

        text = open(PROMPT, encoding="utf-8").read()
        results = {}

        def req(tag, seq_id, prompt):
            body = json.dumps({
                "prompt": prompt, "n_predict": int(a.n), "temperature": 0.0,
                "seed": 42, "id_slot": seq_id, "cache_prompt": False,
                "stream": False,
            }).encode()
            r = urllib.request.Request(base + "/completion", data=body,
                                       headers={"Content-Type": "application/json"})
            t0 = time.time()
            d = json.loads(urllib.request.urlopen(r, timeout=1800).read())
            dt = time.time() - t0
            c = d["content"]
            tim = d.get("timings", {})
            results[tag] = {
                "chars": len(c),
                "sha": hashlib.sha256(c.encode()).hexdigest()[:12],
                "wall_s": round(dt, 2),
                "predicted_per_s": round(tim.get("predicted_per_second", 0.0), 2),
                "prompt_per_s": round(tim.get("prompt_per_second", 0.0), 2),
            }

        ts = [threading.Thread(target=req, args=("slotA", 0, text)),
              threading.Thread(target=req, args=("slotB", 1, text))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        srv.terminate()
        try:
            srv.wait(timeout=60)
        except subprocess.TimeoutExpired:
            srv.kill()

    s = open(log, errors="replace").read()
    acc = re.findall(r"id\s+(\d+) \| task.*?draft acceptance = ([0-9.]+)", s)
    reuse = re.findall(r"DRAFT_GRAPH.*slot_alt=(\d+) reused=(\d+)", s)
    nr = {}
    for slot, r in reuse:
        nr.setdefault(slot, []).append(int(r))
    reuse_pct = {k: round(100.0 * sum(v) / len(v), 1) for k, v in nr.items() if v}
    print("LIB=%s  results=%s" % (a.lib, json.dumps(results, indent=None)))
    print("LIB=%s  draft acceptance per slot=%s  draft_ubatches=%s  reuse%%=%s"
          % (a.lib, acc, {k: len(v) for k, v in nr.items()}, reuse_pct))
    return 0


if __name__ == "__main__":
    sys.exit(main())
