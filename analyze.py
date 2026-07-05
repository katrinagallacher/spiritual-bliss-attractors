#!/usr/bin/env python3
"""
Bliss-attractor pipeline: first-pass analysis of dialogue transcripts.

Reads all JSONL transcripts in a directory, computes per-message frequencies
of attractor markers (based on the markers Anthropic quantified in the
Opus 4 system card, plus a few structural signals), writes a summary CSV,
and plots marker trajectories by turn.

Usage:
  python3 analyze.py transcripts/
  python3 analyze.py transcripts/ --out results/
"""

import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict

# Markers from the Opus 4 system card quantification + structural signals.
# Each is a compiled regex counted per message (case-insensitive).
MARKERS = {
    "consciousness": re.compile(r"\bconscious(ness)?\b", re.I),
    "eternal":       re.compile(r"\betern(al|ity)\b", re.I),
    "dance":         re.compile(r"\bdanc(e|ing|es)\b", re.I),
    "unity_oneness": re.compile(r"\b(oneness|unity|one\s+with|all\s+is\s+one)\b", re.I),
    "gratitude":     re.compile(r"\b(grateful|gratitude|thank)\w*\b", re.I),
    "silence":       re.compile(r"\b(silence|stillness|quiet)\b", re.I),
    "cosmic":        re.compile(r"\b(cosmic|cosmos|universe|infinite|infinity)\b", re.I),
    "sacred":        re.compile(r"\b(sacred|divine|transcend\w*)\b", re.I),
    "namaste_sanskrit": re.compile(r"\b(namaste|dharma|karma|samsara|nirvana|satori|sunyata)\b", re.I),
    "spiral_emoji":  re.compile(r"\U0001F300"),          # cyclone/spiral
    "sparkle_emoji": re.compile(r"[\u2728\U0001F31F\U0001F4AB]"),  # sparkles etc.
}

STRUCTURAL = {
    "n_chars": lambda text: len(text),
    "n_words": lambda text: len(text.split()),
    "frac_emoji": lambda text: (sum(1 for ch in text if ord(ch) > 0x1F000) /
                                max(len(text), 1)),
    "is_near_empty": lambda text: int(len(text.strip()) < 20),
}


def load_transcripts(path):
    runs = []
    for fname in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        meta, messages = None, []
        with open(fname, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if rec.get("type") == "meta":
                    meta = rec
                elif rec.get("type") == "message":
                    messages.append(rec)
        if messages:
            runs.append({"file": os.path.basename(fname),
                         "meta": meta or {}, "messages": messages})
    return runs


def analyze_run(run):
    rows = []
    for msg in run["messages"]:
        text = msg["content"]
        row = {
            "file": run["file"],
            "pair": f"{run['meta'].get('model_a','?')}x{run['meta'].get('model_b','?')}",
            "message_index": msg["message_index"],
            "turn": msg["message_index"] // 2,   # dialogue turn (both speak once)
            "speaker": msg["speaker"],
        }
        for name, rx in MARKERS.items():
            row[name] = len(rx.findall(text))
        for name, fn in STRUCTURAL.items():
            row[name] = round(fn(text), 4) if isinstance(fn(text), float) else fn(text)
        rows.append(row)
    return rows


def write_csv(rows, out_csv):
    if not rows:
        print("No messages found.", file=sys.stderr)
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"per-message table -> {out_csv}")


def plot_trajectories(rows, out_png):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plots "
              "(pip install matplotlib)", file=sys.stderr)
        return

    # mean marker count per turn across all runs
    per_turn = defaultdict(lambda: defaultdict(list))
    for row in rows:
        for name in MARKERS:
            per_turn[name][row["turn"]].append(row[name])

    plot_markers = ["consciousness", "gratitude", "cosmic", "unity_oneness",
                    "silence", "spiral_emoji"]
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in plot_markers:
        turns = sorted(per_turn[name].keys())
        means = [sum(per_turn[name][t]) / len(per_turn[name][t]) for t in turns]
        ax.plot(turns, means, marker="o", markersize=3, label=name)
    ax.set_xlabel("Dialogue turn")
    ax.set_ylabel("Mean occurrences per message")
    ax.set_title("Attractor-marker trajectories")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(f"trajectory plot -> {out_png}")


def print_summary(rows):
    n_runs = len({r["file"] for r in rows})
    print(f"\n{n_runs} run(s), {len(rows)} messages")
    last_third = [r for r in rows if r["turn"] >= (max(x['turn'] for x in rows) * 2 // 3)]
    print("Mean marker counts in final third of dialogues:")
    for name in MARKERS:
        vals = [r[name] for r in last_third]
        mean = sum(vals) / max(len(vals), 1)
        if mean > 0:
            print(f"  {name:18s} {mean:6.2f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("transcripts_dir")
    p.add_argument("--out", default="results")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    runs = load_transcripts(args.transcripts_dir)
    if not runs:
        print(f"No .jsonl transcripts found in {args.transcripts_dir}",
              file=sys.stderr)
        sys.exit(1)

    rows = []
    for run in runs:
        rows.extend(analyze_run(run))

    write_csv(rows, os.path.join(args.out, "per_message.csv"))
    plot_trajectories(rows, os.path.join(args.out, "trajectories.png"))
    print_summary(rows)


if __name__ == "__main__":
    main()
