from open_mythos.main import MythosConfig

# Parameter budget breakdown per variant:
#   total ≈ embed + prelude/coda dense blocks + recurrent MLA + MoE
#   MoE   = 3 * dim * expert_dim * (n_experts + n_shared * n_experts_per_tok)
# expert_dim is solved from the residual budget after all other terms.


def from_hf_config(hf_cfg) -> MythosConfig:
    """Build a MythosConfig that dimensionally matches a HF model config.

    Works with Llama/Qwen/Mistral style configs.  Forces ``attn_type='gqa'``
    because standard HF models do not implement MLA.

    Args:
        hf_cfg: a ``transformers.PretrainedConfig`` instance or a plain dict.

    Returns:
        ``MythosConfig`` with compatible dimensions.
    """
    # Qwen3.5 nests text config inside a vision wrapper
    if hasattr(hf_cfg, "text_config"):
        c = hf_cfg.text_config
    else:
        c = hf_cfg

    hidden_size = getattr(c, "hidden_size", 2048)
    num_layers = getattr(c, "num_hidden_layers", 24)
    n_heads = getattr(c, "num_attention_heads", 16)
    n_kv_heads = getattr(c, "num_key_value_heads", n_heads)
    vocab_size = getattr(c, "vocab_size", 32000)
    max_seq_len = getattr(c, "max_position_embeddings", 4096)
    intermediate_size = getattr(c, "intermediate_size", hidden_size * 4 // 3)
    rope_theta = getattr(c, "rope_theta", 500000.0)

    # Split layers roughly: 1/4 prelude, 1/2 recurrent loop budget, 1/4 coda
    prelude = max(2, num_layers // 4)
    coda = max(2, num_layers // 4)
    loop_iters = max(4, num_layers - prelude - coda)

    # MoE defaults scaled to model size
    n_experts = 64 if hidden_size <= 4096 else 128 if hidden_size <= 8192 else 256
    n_shared = 2 if n_experts <= 64 else 4
    topk = 4 if n_experts <= 64 else 8

    return MythosConfig(
        vocab_size=vocab_size,
        dim=hidden_size,
        n_heads=n_heads,
        n_kv_heads=n_kv_heads,
        max_seq_len=max_seq_len,
        max_loop_iters=loop_iters,
        prelude_layers=prelude,
        coda_layers=coda,
        attn_type="gqa",
        # MLA params ignored because attn_type='gqa'
        kv_lora_rank=0,
        q_lora_rank=0,
        qk_rope_head_dim=0,
        qk_nope_head_dim=0,
        v_head_dim=0,
        n_experts=n_experts,
        n_shared_experts=n_shared,
        n_experts_per_tok=topk,
        intermediate_size=intermediate_size,
        expert_dim=intermediate_size // topk,
        act_threshold=0.99,
        rope_theta=rope_theta,
        lora_rank=16 if hidden_size <= 4096 else 32,
        max_output_tokens=min(max_seq_len, 131072),
    )


def mythos_1b() -> MythosConfig:
    """1B parameter config. Small research/fine-tuning model. dim=2048, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=2048,
        n_heads=16,
        n_kv_heads=4,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=256,
        q_lora_rank=512,
        qk_rope_head_dim=32,
        qk_nope_head_dim=64,
        v_head_dim=64,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=2048,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_3b() -> MythosConfig:
    """3B parameter config. Compact inference model. dim=3072, 64 experts, 16 loop iters, 4k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=3072,
        n_heads=24,
        n_kv_heads=6,
        max_seq_len=4096,
        max_loop_iters=16,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=384,
        q_lora_rank=768,
        qk_rope_head_dim=32,
        qk_nope_head_dim=96,
        v_head_dim=96,
        n_experts=64,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=4096,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=8,
    )


def mythos_10b() -> MythosConfig:
    """10B parameter config. Mid-scale general model. dim=4096, 128 experts, 24 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=4096,
        n_heads=32,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=24,
        prelude_layers=2,
        coda_layers=2,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=128,
        n_shared_experts=2,
        n_experts_per_tok=4,
        expert_dim=5632,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=16,
    )


def mythos_50b() -> MythosConfig:
    """50B parameter config. Large reasoning model. dim=6144, 256 experts, 32 loop iters, 8k context."""
    return MythosConfig(
        vocab_size=32000,
        dim=6144,
        n_heads=48,
        n_kv_heads=8,
        max_seq_len=8192,
        max_loop_iters=32,
        prelude_layers=3,
        coda_layers=3,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=1536,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=4,
        expert_dim=9728,
        act_threshold=0.99,
        rope_theta=500000.0,
        lora_rank=32,
    )


def mythos_100b() -> MythosConfig:
    """100B parameter config. Frontier-class model. dim=8192, 256 experts, 32 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=32000,
        dim=8192,
        n_heads=64,
        n_kv_heads=8,
        max_seq_len=1000000,
        max_loop_iters=32,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=512,
        q_lora_rank=2048,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=256,
        n_shared_experts=4,
        n_experts_per_tok=8,
        expert_dim=13568,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=64,
        max_output_tokens=131072,
    )


def mythos_500b() -> MythosConfig:
    """500B parameter config. Ultra-scale MoE model. dim=12288, 512 experts, 48 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=12288,
        n_heads=96,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=48,
        prelude_layers=4,
        coda_layers=4,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=3072,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=23040,
        act_threshold=0.99,
        rope_theta=1000000.0,
        lora_rank=128,
        max_output_tokens=131072,
    )


def mythos_1t() -> MythosConfig:
    """1T parameter config. Maximum scale. dim=16384, 512 experts, 64 loop iters, 1M context, 128k output."""
    return MythosConfig(
        vocab_size=100000,
        dim=16384,
        n_heads=128,
        n_kv_heads=16,
        max_seq_len=1000000,
        max_loop_iters=64,
        prelude_layers=6,
        coda_layers=6,
        attn_type="mla",
        kv_lora_rank=1024,
        q_lora_rank=4096,
        qk_rope_head_dim=64,
        qk_nope_head_dim=128,
        v_head_dim=128,
        n_experts=512,
        n_shared_experts=8,
        n_experts_per_tok=8,
        expert_dim=34560,
        act_threshold=0.99,
        rope_theta=2000000.0,
        lora_rank=256,
        max_output_tokens=131072,
    )
