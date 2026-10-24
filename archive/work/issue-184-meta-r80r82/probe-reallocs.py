#!/usr/bin/env python3
"""How often does a config actually re-allocate a graph?  (TODO #80, retired 2026-10-24.)

The split-state pre-warm walk is paid per graph re-allocation, so its cost is a function of the
re-allocation rate.  llama-server prints that rate per request ("graphs reused" / "graph reallocs"
in `slot print_timing`).  This probe starts a 4-slot server, warms every slot's prompt cache
serially (1 token each) so the measured window is pure decode, fires NP concurrent greedy
completions, and prints the counters.

  python3 probe-reallocs.py dense4b      # Qwen3.5-4B Q8_0, 2 GPU tensor
  python3 probe-reallocs.py flashnext    # the reporter's shape (Flash-Next IQ3_XXS + MTP)

Env overrides: NP, NPRED, DEVICES (HIP_VISIBLE_DEVICES), PROMPT_LEN, PORT, BUILD (llama.cpp tree).
"""
import json, os, subprocess, sys, threading, time, urllib.request

SHAPES = {
    "dense4b": {
        "model": "/llm/models/Qwen3.5/4B/Q8_0/Qwen3.5-4B-Q8_0.gguf",
        "args":  ["-sm", "tensor", "-c", "16384", "-b", "2048", "-ub", "2048"],
        "npred": 128, "plen": 1500, "devices": "0,1", "port": 8093,
    },
    "flashnext": {
        # issue #184 / TODO #80's own shape: the one whose ~230 re-allocations per run made the walk
        # worth the reporter's attention.
        "model": "/llm/models/Qwen3.8/Flash-Next/GSQ-IQ3_XXS/Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf",
        "args":  ["-md", "/llm/models/Qwen3.8/Flash-Next/IQ4_NL/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf",
                  "-ngld", "99", "-ncmoe", "47", "-sm", "tensor", "-fa", "auto", "-ctk", "q8_0",
                  "-ctv", "q8_0", "-c", "131072", "-b", "2048", "-ub", "2048",
                  "--load-mode", "none", "--lazy-mode", "off",
                  "--spec-type", "draft-mtp-adaptive", "--spec-draft-n-max", "3"],
        "npred": 192, "plen": 2000, "devices": "0,1", "port": 8094,
    },
}

shape = SHAPES[sys.argv[1] if len(sys.argv) > 1 else "flashnext"]
BUILD = os.environ.get("BUILD", os.path.expanduser("~/llama.cpp/build-rocm/bin"))
SERVER = os.path.join(BUILD, "llama-server")
PORT = int(os.environ.get("PORT", shape["port"]))
NP = int(os.environ.get("NP", "4"))
NPRED = int(os.environ.get("NPRED", shape["npred"]))
PROMPT = open(os.path.expanduser("~/llama-cpp-rdna-boosts/prompts/prose-rdna-boosts.txt")).read()[:int(os.environ.get("PROMPT_LEN", shape["plen"]))]
LOG = os.environ.get("LOG", f"/tmp/probe-reallocs-{sys.argv[1] if len(sys.argv) > 1 else 'flashnext'}.log")


def env():
    e = os.environ.copy()
    e.update({"LD_LIBRARY_PATH": "/opt/rocm-10.1.0-gfx120X/lib",
              "HIP_VISIBLE_DEVICES": os.environ.get("DEVICES", shape["devices"])})
    return e


def args():
    return ([SERVER, "-m", shape["model"], "-ngl", "99"] + shape["args"] +
            ["-t", "8", "-np", str(NP), "--fit", "off",
             "--host", "127.0.0.1", "--port", str(PORT)])


def health(proc, t=900):
    t0 = time.time()
    while time.time() - t0 < t:
        if proc.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=5) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def post(payload, t=900):
    r = urllib.request.Request(f"http://127.0.0.1:{PORT}/completion",
                               data=json.dumps(payload).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=t) as resp:
        return json.load(resp)


f = open(LOG, "w")
proc = subprocess.Popen(args(), env=env(), stdout=f, stderr=subprocess.STDOUT)
try:
    if not health(proc):
        print("SERVER FAILED", LOG)
        sys.exit(1)
    for s in range(NP):
        post({"prompt": PROMPT, "n_predict": 1, "temperature": 0, "cache_prompt": True, "id_slot": s})
    time.sleep(1.0)
    out = [None] * NP

    def go(s, i):
        out[i] = post({"prompt": PROMPT, "n_predict": NPRED, "temperature": 0,
                       "cache_prompt": True, "id_slot": s})

    th = [threading.Thread(target=go, args=(s, i)) for i, s in enumerate(range(NP))]
    t0 = time.time()
    for t in th:
        t.start()
    for t in th:
        t.join()
    wall = time.time() - t0
finally:
    proc.terminate()
    try:
        proc.wait(30)
    except Exception:
        proc.kill()
    f.close()

time.sleep(1.0)
print(f"{wall:.2f}s for {NP} streams x {NPRED} tokens   (log {LOG})")
for line in open(LOG, errors="replace").read().splitlines():
    if "graphs reused" in line or "graph reallocs" in line:
        print(line)
