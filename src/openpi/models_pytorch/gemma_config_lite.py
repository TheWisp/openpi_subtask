"""
Minimal configs for PyTorch inference — no JAX/Flax dependency.

Mirrors Pi0Config and gemma get_config() for gemma_2b and gemma_300m variants.
"""
from dataclasses import dataclass, field


@dataclass
class GemmaConfigLite:
    width: int
    depth: int
    mlp_dim: int
    num_heads: int
    head_dim: int
    num_kv_heads: int


_CONFIGS = {
    "gemma_2b": GemmaConfigLite(
        width=2048, depth=18, mlp_dim=16384, num_heads=8, head_dim=256, num_kv_heads=1
    ),
    "gemma_2b_lora": GemmaConfigLite(
        width=2048, depth=18, mlp_dim=16384, num_heads=8, head_dim=256, num_kv_heads=1
    ),
    "gemma_300m": GemmaConfigLite(
        width=1024, depth=18, mlp_dim=4096, num_heads=8, head_dim=256, num_kv_heads=1
    ),
    "gemma_300m_lora": GemmaConfigLite(
        width=1024, depth=18, mlp_dim=4096, num_heads=8, head_dim=256, num_kv_heads=1
    ),
    "dummy": GemmaConfigLite(
        width=64, depth=2, mlp_dim=128, num_heads=2, head_dim=32, num_kv_heads=1
    ),
}


def get_config(variant: str) -> GemmaConfigLite:
    if variant not in _CONFIGS:
        raise ValueError(f"Unknown gemma variant: {variant!r}. Available: {list(_CONFIGS)}")
    return _CONFIGS[variant]


@dataclass
class Pi0ConfigLite:
    """Minimal Pi0Config for PyTorch inference — no JAX/Flax dependency."""
    action_dim: int = 32
    action_horizon: int = 50
    paligemma_variant: str = "gemma_2b"
    action_expert_variant: str = "gemma_300m"
    pi05: bool = True
    dtype: str = "bfloat16"
    max_token_len: int = 200
    discrete_state_input: bool = True
