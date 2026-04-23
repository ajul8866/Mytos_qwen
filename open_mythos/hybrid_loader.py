"""
Hybrid loader — map HuggingFace pretrained weights into OpenMythos Prelude/Coda.

Loads weights DIRECTLY to GPU via device_map="cuda". CPU RAM is kept minimal.
"""
from __future__ import annotations

import gc
from typing import Optional

import torch
from transformers import AutoModelForCausalLM


def _map_block(hf_state, prefix, block, device):
    block.attn.wq.weight.data.copy_(hf_state[f"{prefix}.self_attn.q_proj.weight"].to(device))
    block.attn.wk.weight.data.copy_(hf_state[f"{prefix}.self_attn.k_proj.weight"].to(device))
    block.attn.wv.weight.data.copy_(hf_state[f"{prefix}.self_attn.v_proj.weight"].to(device))
    block.attn.wo.weight.data.copy_(hf_state[f"{prefix}.self_attn.o_proj.weight"].to(device))
    block.ffn.gate.weight.data.copy_(hf_state[f"{prefix}.mlp.gate_proj.weight"].to(device))
    block.ffn.up.weight.data.copy_(hf_state[f"{prefix}.mlp.up_proj.weight"].to(device))
    block.ffn.down.weight.data.copy_(hf_state[f"{prefix}.mlp.down_proj.weight"].to(device))
    block.attn_norm.weight.data.copy_(hf_state[f"{prefix}.input_layernorm.weight"].to(device))
    block.ffn_norm.weight.data.copy_(hf_state[f"{prefix}.post_attention_layernorm.weight"].to(device))


def load_hf_weights(model, hf_model_id: str, dtype: Optional[torch.dtype] = None):
    if model.cfg.attn_type != "gqa":
        raise ValueError("Hybrid loading requires attn_type='gqa'.")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Model must be on CUDA. Call model.to('cuda') first.")

    print(f"[Hybrid] Loading HF weights DIRECTLY to {device}: {hf_model_id}")

    hf_model = AutoModelForCausalLM.from_pretrained(
        hf_model_id,
        torch_dtype=dtype or next(model.parameters()).dtype,
        device_map={"": device},         # <<< force EVERY tensor onto target CUDA
        low_cpu_mem_usage=True,           # <<< prevents CPU staging
        offload_state_dict=False,         # <<< never offload to CPU/disk
    )
    gc.collect()

    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(hf_model_id)
    c = getattr(cfg, "text_config", cfg)
    total_layers = c.num_hidden_layers
    layer_types = getattr(c, "layer_types", None)

    n_prelude = len(model.prelude)
    n_coda = len(model.coda)

    # --- embed ---
    rows = min(hf_model.model.embed_tokens.weight.shape[0], model.embed.weight.shape[0])
    model.embed.weight.data[:rows].copy_(hf_model.model.embed_tokens.weight[:rows])

    # --- prelude ---
    for i in range(n_prelude):
        lt = layer_types[i] if layer_types else "full_attention"
        _map_block(hf_model.model.layers[i].state_dict(), f"model.layers.{i}", model.prelude[i], device)
        if lt != "full_attention":
            print(f"[Hybrid] Prelude layer {i} is '{lt}'; attention weights loaded anyway (FFN+norm used)")

    # --- coda ---
    for j in range(n_coda):
        idx = total_layers - n_coda + j
        lt = layer_types[idx] if layer_types else "full_attention"
        _map_block(hf_model.model.layers[idx].state_dict(), f"model.layers.{idx}", model.coda[j], device)
        if lt != "full_attention":
            print(f"[Hybrid] Coda layer {j} is '{lt}'; attention weights loaded anyway")

    # --- norm & head ---
    model.norm.weight.data.copy_(hf_model.model.norm.weight)
    if model.head.weight.shape == hf_model.lm_head.weight.shape:
        model.head.weight.data.copy_(hf_model.lm_head.weight)

    del hf_model
    gc.collect()
    torch.cuda.empty_cache()
    print(f"[Hybrid] Done. RecurrentBlock remains randomly initialised.")


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
