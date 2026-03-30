#!/usr/bin/env python3
"""
Convert soarm-pi05-fast-7998 JAX checkpoint to PyTorch safetensors.

Same architecture as soarm-pi05-state-11997 (Pi0.5), but trained with FAST tokens.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent))  # for examples module

import os
os.environ["JAX_ENABLE_X64"] = "false"

from openpi.models.pi0_config import Pi0Config
from examples.convert_jax_model_to_pytorch import convert_pi0_checkpoint

CHECKPOINT_DIR = str(Path.home() / ".cache/openpi/checkpoints/soarm-pi05-fast-7998")
OUTPUT_PATH = str(Path.home() / ".cache/lerobot/converted/soarm-pi05-fast-7998-pytorch")

model_config = Pi0Config(
    action_horizon=50,
    action_dim=32,
    paligemma_variant="gemma_2b",  # full-rank (LoRA merged in JAX checkpoint)
    pi05=True,
)

print(f"Converting: {CHECKPOINT_DIR}")
print(f"Output:     {OUTPUT_PATH}")
print(f"Config:     pi05={model_config.pi05}, action_horizon={model_config.action_horizon}, action_dim={model_config.action_dim}")

convert_pi0_checkpoint(
    checkpoint_dir=CHECKPOINT_DIR,
    precision="bfloat16",
    output_path=OUTPUT_PATH,
    model_config=model_config,
)

# Fix tied weights
import safetensors.torch as st

output_file = str(Path(OUTPUT_PATH) / "model.safetensors")
print("Post-processing: adding embed_tokens key for tied weights...")
sd = st.load_file(output_file)
lm_head_key = "paligemma_with_expert.paligemma.lm_head.weight"
embed_key = "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight"
if lm_head_key in sd and embed_key not in sd:
    sd[embed_key] = sd[lm_head_key].clone()
    st.save_file(sd, output_file)
    print(f"  Added {embed_key}")
else:
    print(f"  Already present or lm_head missing: lm_head={lm_head_key in sd}, embed={embed_key in sd}")
print("Done.")
