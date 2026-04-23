"""
Hybrid loader — map HuggingFace pretrained weights into OpenMythos Prelude/Coda.

GPU-ONLY loading: weights are streamed directly from safetensors files to CUDA
without staging the full HF model in CPU RAM. Required for large models that
do not fit in host memory.

Usage
-----
    from open_mythos.hybrid_loader import load_hf_weights, freeze_pretrained_layers
    model = OpenMythos(cfg).to("cuda")
    load_hf_weights(model, "sulpikar2/Qwen3.6-27B-hereticv3")
"""

from __future__ import annotations

import json
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file


def _get_tensor(state_dict: dict, key: str, device: torch.device) -> torch.Tensor:
    if key not in state_dict:
        raise KeyError(f"Missing expected weight in HF checkpoint: {key}")
    t = state_dict[key]
    if t.device != device:
        t = t.to(device, non_blocking=True)
    return t


def _map_block_full(hf_state: dict, prefix: str, block, loaded: dict, device: torch.device):
    block.attn.wq.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.self_attn.q_proj.weight", device))
    block.attn.wk.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.self_attn.k_proj.weight", device))
    block.attn.wv.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.self_attn.v_proj.weight", device))
    block.attn.wo.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.self_attn.o_proj.weight", device))
    block.ffn.gate.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.mlp.gate_proj.weight", device))
    block.ffn.up.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.mlp.up_proj.weight", device))
    block.ffn.down.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.mlp.down_proj.weight", device))
    block.attn_norm.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.input_layernorm.weight", device))
    block.ffn_norm.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.post_attention_layernorm.weight", device))
    loaded[f"{prefix}.full"] = True


def _map_block_ffn(hf_state: dict, prefix: str, block, loaded: dict, device: torch.device):
    block.ffn.gate.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.mlp.gate_proj.weight", device))
    block.ffn.up.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.mlp.up_proj.weight", device))
    block.ffn.down.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.mlp.down_proj.weight", device))
    block.attn_norm.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.input_layernorm.weight", device))
    block.ffn_norm.weight.data.copy_(_get_tensor(hf_state, f"{prefix}.post_attention_layernorm.weight", device))
    loaded[f"{prefix}.ffn"] = True


def load_hf_weights(model, hf_model_id: str, dtype: Optional[torch.dtype] = None):
    """Load pretrained HF weights DIRECTLY to GPU without CPU staging.

    Args:
        model: OpenMythos instance (must have attn_type='gqa', already on CUDA).
        hf_model_id: HuggingFace model ID.
        dtype: Optional dtype cast (default: keep original dtype).
    """
    if model.cfg.attn_type != "gqa":
        raise ValueError("Hybrid loading requires attn_type='gqa'.")

    device = next(model.parameters()).device
    if device.type != "cuda":
        raise RuntimeError("Model must be on CUDA before calling load_hf_weights. Call model.to('cuda') first.")

    print(f"[Hybrid] Loading HF weights DIRECTLY to {device}: {hf_model_id}")

    from transformers import AutoConfig
    hf_cfg = AutoConfig.from_pretrained(hf_model_id)
    if hasattr(hf_cfg, "text_config"):
        c = hf_cfg.text_config
    else:
        c = hf_cfg

    total_layers = c.num_hidden_layers
    layer_types = getattr(c, "layer_types", None)
    print(f"[Hybrid] HF layers={total_layers}, device={device}")

    try:
        index_path = hf_hub_download(hf_model_id, filename="model.safetensors.index.json")
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
    except Exception:
        weight_map = {}

    if weight_map:
        needed_files = set()
        for key in weight_map:
            needed_files.add(weight_map[key])
    else:
        needed_files = {"model.safetensors"}

    print(f"[Hybrid] Will load {len(needed_files)} safetensors shard(s)")

    hf_state: dict[str, torch.Tensor] = {}
    for filename in needed_files:
        shard_path = hf_hub_download(hf_model_id, filename=filename)
        print(f"[Hybrid] Loading shard: {filename} -> {device}")
        shard = load_file(shard_path, device=str(device))
        hf_state.update(shard)
        del shard

    n_prelude = len(model.prelude)
    n_coda = len(model.coda)
    loaded_flags: dict = {}

    embed_key = "model.embed_tokens.weight"
    if embed_key in hf_state:
        hf_vocab, hf_dim = hf_state[embed_key].shape
        my_vocab, my_dim = model.embed.weight.shape
        if hf_dim != my_dim:
            raise ValueError(f"Embed dim mismatch: HF={hf_dim}, OpenMythos={my_dim}")
        rows = min(hf_vocab, my_vocab)
        model.embed.weight.data[:rows].copy_(hf_state[embed_key][:rows])
        if hf_vocab != my_vocab:
            print(f"[Hybrid] WARNING vocab mismatch HF={hf_vocab} vs Mythos={my_vocab}; copied {rows} rows.")

    for i in range(n_prelude):
        prefix = f"model.layers.{i}"
        lt = layer_types[i] if layer_types else "full_attention"
        if lt == "full_attention":
            _map_block_full(hf_state, prefix, model.prelude[i], loaded_flags, device)
        else:
            print(f"[Hybrid] Prelude layer {i} is '{lt}'; loading FFN+norm only, attention kept random.")
            _map_block_ffn(hf_state, prefix, model.prelude[i], loaded_flags, device)

    for j in range(n_coda):
        hf_idx = total_layers - n_coda + j
        prefix = f"model.layers.{hf_idx}"
        lt = layer_types[hf_idx] if layer_types else "full_attention"
        if lt == "full_attention":
            _map_block_full(hf_state, prefix, model.coda[j], loaded_flags, device)
        else:
            print(f"[Hybrid] Coda layer {j} (HF idx {hf_idx}) is '{lt}'; loading FFN+norm only, attention kept random.")
            _map_block_ffn(hf_state, prefix, model.coda[j], loaded_flags, device)

    if "model.norm.weight" in hf_state:
        model.norm.weight.data.copy_(hf_state["model.norm.weight"])

    if "lm_head.weight" in hf_state:
        hf_v, hf_d = hf_state["lm_head.weight"].shape
        my_v, my_d = model.head.weight.shape
        if hf_d == my_d and hf_v == my_v:
            model.head.weight.data.copy_(hf_state["lm_head.weight"])
        else:
            print(f"[Hybrid] lm_head shape mismatch ({hf_v},{hf_d}) vs ({my_v},{my_d}); skipped.")

    print(f"[Hybrid] Loaded {len(loaded_flags)} HF sub-layers into OpenMythos on {device}.")
    del hf_state
    torch.cuda.empty_cache()


def freeze_pretrained_layers(
    model,
    freeze_embed: bool = True,
    freeze_prelude: bool = True,
    freeze_coda: bool = True,
    freeze_norm: bool = True,
    freeze_head: bool = False,
):
    """Freeze layers carrying pretrained HF weights; leave RecurrentBlock trainable."""
    trainable = 0
    frozen = 0
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

        if should_freeze:
            param.requires_grad = False
            frozen += param.numel()
        else:
            param.requires_grad = True
            trainable += param.numel()

    total = trainable + frozen
    print(f"[Freeze] Frozen {frozen:,} ({frozen/total:.1%}) | Trainable {trainable:,} ({trainable/total:.1%})")
    print(f"[Freeze] RecurrentBlock trainable: {any(p.requires_grad for p in model.recurrent.parameters())}")
