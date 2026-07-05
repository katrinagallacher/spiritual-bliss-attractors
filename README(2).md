# Bliss-Attractor Pipeline

Model x model self-dialogue runner + first-pass analysis, for investigating
whether the "spiritual bliss attractor" (Anthropic, Opus 4 system card, 2025)
is intrinsic to LLMs or specific to certain training regimes.

## Files
- `run_dialogues.py` — pairs two models in open-ended dialogue, logs JSONL.
  Backends: local Ollama (default) or any OpenAI-compatible API.
- `analyze.py` — per-turn marker frequencies (consciousness, gratitude,
  unity, silence, spiral emojis, ...), summary CSV, trajectory plot.

## Quick start (local, free)
1. Install Ollama: https://ollama.com/download
2. `ollama pull qwen2.5:7b`
3. `python3 run_dialogues.py --runs 3 --turns 30`
4. `python3 analyze.py transcripts/`
   (plots need `pip install matplotlib`)

## Key parameters
- `--turns`      turns per model (total messages = 2x)
- `--temperature` sampling temperature (default 1.0)
- `--max-tokens` cap per message (default 350)
- `--num-ctx`    Ollama context window (default 32768; lower if RAM-bound —
                 NB: Ollama's own default of ~4k silently truncates long
                 dialogues, which is why this script sets it explicitly)
- `--seed N`     reproducible runs (per-message seeds derived from N)

## Cross-model / API runs
Same script, e.g. OpenRouter:
  python3 run_dialogues.py --backend openai \
    --base-url https://openrouter.ai/api \
    --api-key $LLM_API_KEY \
    --model-a meta-llama/llama-3.1-8b-instruct \
    --model-b qwen/qwen-2.5-7b-instruct

## Design notes
- The transcript is stored once, from model A's point of view; role labels
  are flipped when constructing model B's view, so each model always sees
  itself as "assistant".
- The opener is deliberately minimal and non-leading (no mention of
  consciousness, spirituality, or the attractor).
- Marker set follows the terms Anthropic quantified (consciousness,
  eternal, dance, spiral emoji) plus unity/gratitude/silence/Sanskrit
  and structural signals (message length, emoji fraction, near-empty
  messages as a "silence" proxy).
