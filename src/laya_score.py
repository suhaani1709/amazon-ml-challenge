"""Score record pairs with the team's fine-tuned Laya matcher on Apple GPU (MPS).

Mirrors laya/laya_finetuned/local_pipeline/laya_matcher.py exactly (same base weights,
LoRA wrapping, fine-tuned delta, question and balanced record serialiser); the only change
is the device: their loader requires CUDA, this runs on MPS with fp16 autocast.

    python -m src.laya_score check                 # sanity check on examples.csv + speed
    python -m src.laya_score bench                 # score bench/val_uncertain_pairs -> bench/laya_scores.parquet
"""
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import polars as pl
import torch

ROOT = Path(__file__).resolve().parents[1]
LAYA = ROOT / "laya" / "laya_finetuned"
sys.path.insert(0, str(LAYA / "local_pipeline"))
sys.path.insert(0, str(LAYA / "local_pipeline" / "vendor"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from laya.agent import Agent  # noqa: E402
from laya.common import collate_items  # noqa: E402
import laya_matcher as lm  # noqa: E402  (item(), load_delta(), QUESTION)

DEV = "mps" if torch.backends.mps.is_available() else "cpu"


def load():
    manifest = json.loads((LAYA / "finetune" / "manifest.json").read_text())
    agent = Agent(str(LAYA / "base"), device=DEV)
    agent.model.encoder.config.reference_compile = False
    for p in agent.model.parameters():
        p.requires_grad_(False)
    from peft import LoraConfig, get_peft_model
    agent.model.encoder = get_peft_model(agent.model.encoder,
        LoraConfig(r=4, lora_alpha=8, target_modules=["Wqkv"], lora_dropout=.05))
    for name, param in agent.model.named_parameters():   # same trainable set as training
        if name.startswith(("head.", "scorer.", "type_emb.")):
            param.requires_grad_(True)
    lm.load_delta(agent, LAYA / "finetune" / "best.safetensors", manifest)
    agent.model.to(DEV).eval()
    return agent, manifest["max_len"]


def score(agent, max_len, left, right, batch=64):
    """left/right: lists of dicts(business_name, business_address, country)."""
    items = [lm.item(agent, a, b, max_len) for a, b in zip(left, right)]
    order = sorted(range(len(items)), key=lambda i: len(items[i]["ids"]))
    out = np.zeros(len(items), dtype=np.float32)
    with torch.inference_mode():
        for s in range(0, len(order), batch):
            idx = order[s:s + batch]
            b = collate_items([[items[i]] for i in idx], agent.tok.pad_token_id)
            inputs = {k: b[k].to(DEV) for k in lm.INPUT_KEYS}
            with torch.autocast(DEV, dtype=torch.float16, enabled=DEV != "cpu"):
                logits, _ = agent.model(**inputs)
            out[idx] = logits.float().softmax(-1)[:, 1].cpu().numpy()
    return out


def rec(name, addr, country):
    return {"business_name": name or "", "business_address": addr or "", "country": country or ""}


def main(mode):
    agent, max_len = load()
    if mode == "check":
        ex = pl.read_csv(LAYA / "examples.csv")
        p = score(agent, max_len, [rec(r["name_a"], r["addr_a"], r["country_a"]) for r in ex.iter_rows(named=True)],
                  [rec(r["name_b"], r["addr_b"], r["country_b"]) for r in ex.iter_rows(named=True)])
        print("examples (same business, different business):", [round(float(x), 4) for x in p])
        b = pl.read_parquet(ROOT / "bench" / "val_uncertain_pairs.parquet").head(2000)
        t = time.time()
        score(agent, max_len, [rec(*r) for r in b.select("name_a", "addr_a", "country").rows()],
              [rec(*r) for r in b.select("name_b", "addr_b", "country").rows()])
        print(f"speed: {2000/(time.time()-t):.0f} pairs/s on {DEV}")
    elif mode == "bench":
        b = pl.read_parquet(ROOT / "bench" / "val_uncertain_pairs.parquet")
        t = time.time()
        p = score(agent, max_len, [rec(*r) for r in b.select("name_a", "addr_a", "country").rows()],
                  [rec(*r) for r in b.select("name_b", "addr_b", "country").rows()])
        pl.DataFrame({"p_new": p}).write_parquet(ROOT / "bench" / "laya_scores.parquet")
        print(f"scored {b.height:,} bench pairs in {time.time()-t:.0f}s")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "check")
