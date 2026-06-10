"""Capture a REAL inference trace on a GPU (the forcing function).

Loads a real HF model, greedy-decodes a prompt, and emits a canonical trace with
real token ids + real config dims (model.config.to_dict()) -> the inference tab
becomes a genuine run (no whitespace/mock fakery). Token text is decoded into
each node's payload. Writes demos/proof-compare/traces/inference.real.json, which
build_all.py picks up automatically.

Run on the GPU box:  python3 run_inference.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
TRACERS = REPO_ROOT / "demos" / "proof-compare" / "tracers"
for _p in (REPO_ROOT, TRACERS):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from inference import trace_inference  # noqa: E402

MODEL_ID = "Qwen/Qwen3-1.7B"
PROMPT = "The capital of France is"
MAX_TOKENS = 12


def main() -> int:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda").eval()
    cfg = model.config.to_dict()   # NOT dict(model.config) -- that drops keys

    @torch.inference_mode()
    def next_token(ids):
        t = torch.tensor([ids], device="cuda")
        return int(model(t).logits[0, -1].argmax().item())

    prompt_ids = tok.encode(PROMPT)
    trace = trace_inference(prompt_ids, next_token, MODEL_ID, cfg, MAX_TOKENS)

    # decode token text into payloads for the viz tooltip. Every node carries
    # the token its forward pass PRODUCED -- including the prefill, whose last
    # position produces the first generated token.
    for ev in trace["events"]:
        text = tok.decode([ev["payload"]["token_id"]])
        ev["payload"]["token"] = text
        if ev["kind"] == "prefill":
            ev["payload"]["prompt"] = PROMPT
            ev["label"] = "prefill"
        else:
            ev["label"] = text.strip() or "·"

    out = REPO_ROOT / "demos" / "proof-compare" / "traces" / "inference.real.json"
    out.write_text(json.dumps(trace))

    decoded = tok.decode([e["payload"]["token_id"] for e in trace["events"]])
    print(f"PROMPT : {PROMPT!r}")
    print(f"OUTPUT : {decoded!r}")
    print(f"model  : {MODEL_ID}  ({cfg['num_hidden_layers']}L d={cfg['hidden_size']})")
    print(f"events : {len(trace['events'])}  -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
