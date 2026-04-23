"""
Hybrid loader — map HuggingFace pretrained weights into OpenMythos Prelude/Coda.

ZERO CPU staging: downloads safetensors shards and loads them DIRECTLY to
CUDA using ``safetensors.torch.load_file(..., device="cuda")``.
No ``AutoModelForCausalLM`` is instantiated, so CPU RAM stays minimal.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _load_shard(filename: str, repo_id: str, device: str) -> dict[str, torch.Tensor]:
    path = hf_hub_download(repo_id, filename=filename, local_files_only=False)
    return load_file(path, device=device)  # <<< loads directly to GPU, zero CPU copy


def load_hf_weights(model, hf_model_id: str, dtype: Optional[torch.dtype] = None):
    if model.cfg.attn_type != "gqa":
        raise ValueError("Hybrid loading requires attn_type='gqa'.")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Model must be on CUDA. Call model.to('cuda') first.")

    dev_str = str(device)
    print(f"[Hybrid] Loading HF weights ZERO-CPU to {dev_str}: {hf_model_id}")

    # --- download index ---
    index_path = hf_hub_download(hf_model_id, filename="model.safetensors.index.json")
    with open(index_path) as f:
        weight_map = json.load(f)["weight_map"]

    # --- determine needed keys ---
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf_model_id)
    c = getattr(cfg, "text_config", cfg)
    total_layers = c.num_hidden_layers
    layer_types = getattr(c, "layer_types", None)

    needed = {"model.embed_tokens.weight", "model.norm.weight", "lm_head.weight"}
    n_prelude = len(model.prelude)
    n_coda = len(model.coda)
    for i in range(n_prelude):
        p = f"model.layers.{i}."
        needed.update(
            f"{p}self_attn.{x}_proj.weight" for x in ["q", "k", "v", "o"]
        )
        needed.update(
            f"{p}mlp.{x}_proj.weight" for x in ["gate", "up", "down"]
        )
        needed.update([f"{p}input_layernorm.weight", f"{p}post_attention_layernorm.weight"])
    for j in range(n_coda):
        idx = total_layers - n_coda + j
        p = f"model.layers.{idx}."
        needed.update(
            f"{p}self_attn.{x}_proj.weight" for x in ["q", "k", "v", "o"]
        )
        needed.update(
            f"{p}mlp.{x}_proj.weight" for x in ["gate", "up", "down"]
        )
        needed.update([f"{p}input_layernorm.weight", f"{p}post_attention_layernorm.weight"])

    needed_files = sorted(set(weight_map[k] for k in needed if k in weight_map))
    print(f"[Hybrid] {len(needed_files)} shard(s) needed")

    # --- load shards one by one, map immediately, then delete ---
    for fname in needed_files:
        print(f"[Hybrid] Loading shard: {fname}")
        shard = _load_shard(fname, hf_model_id, dev_str)

        # embed
        if "model.embed_tokens.weight" in shard:
            hf_v, hf_d = shard["model.embed_tokens.weight"].shape
            my_v, my_d = model.embed.weight.shape
            rows = min(hf_v, my_v)
            model.embed.weight.data[:rows].copy_(shard["model.embed_tokens.weight"][:rows])

        # prelude
        for i in range(n_prelude):
            p = f"model.layers.{i}."
            block = model.prelude[i]
            for x in ["q", "k", "v", "o"]:
                block.attn.wq.weight.data.copy_(shard[f"{p}self_attn.q_proj.weight"]) if x == "q" else None
                block.attn.wk.weight.data.copy_(shard[f"{p}self_attn.k_proj.weight"]) if x == "k" else None
                block.attn.wv.weight.data.copy_(shard[f"{p}self_attn.v_proj.weight"]) if x == "v" else None
                block.attn.wo.weight.data.copy_(shard[f"{p}self_attn.o_proj.weight"]) if x == "o" else None
            block.ffn.gate.weight.data.copy_(shard[f"{p}mlp.gate_proj.weight"])
            block.ffn.up.weight.data.copy_(shard[f"{p}mlp.up_proj.weight"])
            block.ffn.down.weight.data.copy_(shard[f"{p}mlp.down_proj.weight"])
            block.attn_norm.weight.data.copy_(shard[f"{p}input_layernorm.weight"])
            block.ffn_norm.weight.data.copy_(shard[f"{p}post_attention_layernorm.weight"])
            lt = layer_types[i] if layer_types else "full_attention"
            if lt != "full_attention":
                print(f"[Hybrid] Prelude layer {i} is '{lt}' (attention loaded anyway)")

        # coda
        for j in range(n_coda):
            idx = total_layers - n_coda + j
            p = f"model.layers.{idx}."
            block = model.coda[j]
            block.attn.wq.weight.data.copy_(shard[f"{p}self_attn.q_proj.weight"])
            block.attn.wk.weight.data.copy_(shard[f"{p}self_attn.k_proj.weight"])
            block.attn.wv.weight.data.copy_(shard[f"{p}self_attn.v_proj.weight"])
            block.attn.wo.weight.data.copy_(shard[f"{p}self_attn.o_proj.weight"])
            block.ffn.gate.weight.data.copy_(shard[f"{p}mlp.gate_proj.weight"])
            block.ffn.up.weight.data.copy_(shard[f"{p}mlp.up_proj.weight"])
            block.ffn.down.weight.data.copy_(shard[f"{p}mlp.down_proj.weight"])
            block.attn_norm.weight.data.copy_(shard[f"{p}input_layernorm.weight"])
            block.ffn_norm.weight.data.copy_(shard[f"{p}post_attention_layernorm.weight"])
            lt = layer_types[idx] if layer_types else "full_attention"
            if lt != "full_attention":
                print(f"[Hybrid] Coda layer {j} (idx {idx}) is '{lt}'")

        # norm & head
        if "model.norm.weight" in shard:
            model.norm.weight.data.copy_(shard["model.norm.weight"])
        if "lm_head.weight" in shard:
            hf_v, hf_d = shard["lm_head.weight"].shape
            my_v, my_d = model.head.weight.shape
            if hf_d == my_d and hf_v == my_v:
                model.head.weight.data.copy_(shard["lm_head.weight"])

        del shard
        torch.cuda.empty_cache()

    print("[Hybrid] Done. RecurrentBlock remains randomly initialised.")


def freeze_pretrained_layers(
    model,
    freeze_embed: bool = True,
    freeze_prelude: bool = True,
    freeze_coda: bool = True,
    freeze_norm: bool = True,
    freeze_head: bool = False,
):
    trainable = frozen = 0
    for name, param in model.named_parameters():
        should_freeze = False
        if freeze_embed and "embed" in name:
            should_freeze = True
        if freeze_prelude and "prelude" in name:
            should_freeze = True
        if freeze_coda and "coda" in name:
            should_freeze = True
        if freeze_norm and "norm" in name and "recurrent" not in name:
            should_freeze = True
        if freeze_head and "head" in name:
            should_freeze = True

        param.requires_grad = not should_freeze
        if should_freeze:
            frozen += param.numel()
        else:
            trainable += param.numel()

    total = trainable + frozen
    print(f"[Freeze] Frozen {frozen:,} ({frozen/total:.1%}) | Trainable {trainable:,} ({trainable/total:.1%})")
    print(f"[Freeze] RecurrentBlock trainable: {any(p.requires_grad for p in model.recurrent.parameters())}")
