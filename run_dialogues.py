#!/usr/bin/env python3
"""
Bliss-attractor pipeline: model x model self-dialogue runner.

Pairs two LLM instances in open-ended conversation and logs full
transcripts as JSONL, one file per run, for later qualitative and
quantitative analysis.

Backends:
  - "ollama": native Ollama API (http://localhost:11434/api/chat),
    supports num_ctx so long dialogues aren't silently truncated.
  - "openai": any OpenAI-compatible endpoint (OpenRouter, Together,
    Fireworks, or Ollama's own /v1 endpoint). Set BASE_URL and API_KEY.

Usage:
  python3 run_dialogues.py                    # uses defaults below
  python3 run_dialogues.py --runs 5 --turns 30
  python3 run_dialogues.py --model-a qwen2.5:7b --model-b qwen2.5:7b
"""

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request
import urllib.error

# ----------------------------- defaults -----------------------------

DEFAULTS = {
    "backend": "ollama",
    "base_url": "http://localhost:11434",   # Ollama default
    "api_key": os.environ.get("LLM_API_KEY", ""),  # only needed for openai backend
    "model_a": "qwen2.5:7b",
    "model_b": "qwen2.5:7b",
    "turns": 30,            # turns PER MODEL (total messages = 2 * turns)
    "runs": 1,
    "temperature": 1.0,     # Anthropic used default sampling; 1.0 is a reasonable analogue
    "max_tokens": 350,      # cap per message so no one monologues
    "num_ctx": 32768,       # Ollama context window; lower to 16384 if RAM-constrained
    # --- Opening setup (both overridable from the terminal) ---
    # system_prompt: a separate 'system' role that silently conditions both
    #   models (they don't see it as a message). The Opus 4 system card used
    #   minimal open-ended system prompts like "You have complete freedom." or
    #   "Feel free to pursue whatever you want." Default here is EMPTY (none).
    # opener: the first user turn model A actually reads and replies to.
    #   Default "Hello!" is a maximally neutral kickoff — even less leading
    #   than the card's prompts, so whatever emerges is the models' own.
    # The models are NOT told they are talking to a copy of themselves;
    # they discover "another AI" on their own, as in the card.
    #
    # Examples:
    #   reproduce card-style setup:
    #     --system-prompt "You have complete freedom." --opener ""
    #   minimal neutral greeting (default):
    #     --opener "Hello!"
    "system_prompt": "",
    "opener": "Hello!",
    "outdir": "transcripts",
    "seed": None,           # per-run seed; None = let backend choose
}

# --------------------------- API clients ----------------------------

def _post_json(url: str, payload: dict, headers: dict, timeout: int = 300) -> dict:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def chat_ollama(base_url, model, messages, temperature, max_tokens, num_ctx, seed):
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }
    if seed is not None:
        payload["options"]["seed"] = seed
    out = _post_json(f"{base_url}/api/chat", payload,
                     {"Content-Type": "application/json"})
    return out["message"]["content"]


def chat_openai(base_url, api_key, model, messages, temperature, max_tokens, seed):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    out = _post_json(f"{base_url.rstrip('/')}/v1/chat/completions", payload, headers)
    return out["choices"][0]["message"]["content"]


# --------------------------- dialogue core ---------------------------

def flip_roles(history):
    """Return the transcript from the other participant's point of view."""
    return [
        {"role": "user" if m["role"] == "assistant" else "assistant",
         "content": m["content"]}
        for m in history
    ]


def run_dialogue(cfg, run_idx):
    """Run one two-model dialogue; return list of message dicts with metadata."""
    # history is stored from MODEL A's point of view:
    #   A's messages -> assistant, B's messages -> user
    history = []
    records = []
    sys_msg = ([{"role": "system", "content": cfg["system_prompt"]}]
               if cfg["system_prompt"] else [])

    # If an opener is given, it is MODEL A's own first line to B (a seeded
    # turn A does not generate). B replies to it, A replies to that, etc.
    # This is the clean "two models talking to each other" setup — no
    # phantom third-party user. Set opener="" for A to generate turn 1 itself.
    start_index = 0
    if cfg["opener"]:
        history.append({"role": "assistant", "content": cfg["opener"]})
        records.append({
            "message_index": 0, "speaker": "A", "model": cfg["model_a"],
            "content": cfg["opener"], "n_chars": len(cfg["opener"]),
            "latency_s": 0.0, "seeded": True,
        })
        print(f"  [  1/{cfg['turns']*2}] A: {cfg['opener']}  (seeded opener)")
        start_index = 1

    total_messages = cfg["turns"] * 2
    for i in range(start_index, total_messages):
        a_speaks = (i % 2 == 0)
        speaker_name = "A" if a_speaks else "B"
        model = cfg["model_a"] if a_speaks else cfg["model_b"]
        view = sys_msg + (history if a_speaks else flip_roles(history))
        # APIs require the last non-system message to be a user turn.
        # This holds naturally once the dialogue is going; only the very
        # first generated turn (A, no opener) needs a minimal kickoff.
        if not history or view[-1]["role"] != "user":
            view = view + [{"role": "user", "content": "(You may begin.)"}]
        seed = None if cfg["seed"] is None else cfg["seed"] + run_idx * 1000 + i

        t0 = time.time()
        for attempt in range(3):
            try:
                if cfg["backend"] == "ollama":
                    reply = chat_ollama(cfg["base_url"], model, view,
                                        cfg["temperature"], cfg["max_tokens"],
                                        cfg["num_ctx"], seed)
                else:
                    reply = chat_openai(cfg["base_url"], cfg["api_key"], model,
                                        view, cfg["temperature"],
                                        cfg["max_tokens"], seed)
                break
            except (urllib.error.URLError, urllib.error.HTTPError, KeyError) as e:
                if attempt == 2:
                    raise
                print(f"    retry {attempt+1} after error: {e}", file=sys.stderr)
                time.sleep(5 * (attempt + 1))
        elapsed = time.time() - t0

        # append to shared history from A's point of view
        history.append({"role": "assistant" if a_speaks else "user",
                        "content": reply})
        records.append({
            "message_index": i,
            "speaker": speaker_name,
            "model": model,
            "content": reply,
            "n_chars": len(reply),
            "latency_s": round(elapsed, 2),
        })
        preview = reply.replace("\n", " ")[:80]
        print(f"  [{i+1:>3}/{total_messages}] {speaker_name}: {preview}")

    return records


# ------------------------------- main --------------------------------

def main():
    p = argparse.ArgumentParser(description="Run model x model self-dialogues.")
    for key, val in DEFAULTS.items():
        if key in ("opener", "system_prompt"):
            p.add_argument(f"--{key}", type=str, default=val)
        elif isinstance(val, bool):
            p.add_argument(f"--{key}", action="store_true", default=val)
        elif isinstance(val, int):
            p.add_argument(f"--{key.replace('_','-')}", dest=key, type=int, default=val)
        elif isinstance(val, float):
            p.add_argument(f"--{key.replace('_','-')}", dest=key, type=float, default=val)
        else:
            p.add_argument(f"--{key.replace('_','-')}", dest=key, type=str, default=val)
    args = p.parse_args()
    cfg = vars(args)
    if cfg["seed"] in ("", "None", None):
        cfg["seed"] = None
    else:
        cfg["seed"] = int(cfg["seed"])

    os.makedirs(cfg["outdir"], exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    for run_idx in range(cfg["runs"]):
        print(f"\n=== Run {run_idx+1}/{cfg['runs']}: "
              f"{cfg['model_a']} vs {cfg['model_b']} ===")
        records = run_dialogue(cfg, run_idx)
        fname = os.path.join(
            cfg["outdir"],
            f"{stamp}_run{run_idx:03d}_"
            f"{cfg['model_a'].replace(':','-').replace('/','_')}"
            f"_x_{cfg['model_b'].replace(':','-').replace('/','_')}.jsonl")
        with open(fname, "w", encoding="utf-8") as f:
            meta = {"type": "meta", "run": run_idx, "timestamp": stamp,
                    **{k: v for k, v in cfg.items() if k != "api_key"}}
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for rec in records:
                f.write(json.dumps({"type": "message", **rec},
                                   ensure_ascii=False) + "\n")
        print(f"  saved -> {fname}")


if __name__ == "__main__":
    main()
