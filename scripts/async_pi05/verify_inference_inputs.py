#!/usr/bin/env python3
"""Verify inference pipeline inputs match training pipeline.

Dumps the actual model inputs (images, state, tokens) from both the inference
pipeline and the training data pipeline so they can be compared side by side.

Usage:
    # Training pipeline only (no checkpoint needed):
    cd ~/Documents/openpi_subtask
    uv run scripts/async_pi05/verify_inference_inputs.py --skip-inference

    # Both pipelines:
    uv run scripts/async_pi05/verify_inference_inputs.py \
        --checkpoint ~/.cache/openpi/checkpoints/soarm-pi05-flow-lora-8000

    # Inference pipeline only:
    uv run scripts/async_pi05/verify_inference_inputs.py \
        --skip-training \
        --checkpoint ~/.cache/openpi/checkpoints/soarm-pi05-flow-lora-8000

Output: saves comparison data to /tmp/verify_inputs/
"""

import argparse
import json
import logging
import os
import pathlib

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

OUT_DIR = pathlib.Path("/tmp/verify_inputs")


def save_image(img_array, path):
    """Save image array as PNG. Handles [-1,1] float or [0,255] uint8."""
    from PIL import Image
    img = np.asarray(img_array)
    if img.ndim == 4:
        img = img[0]  # remove batch dim
    if np.issubdtype(img.dtype, np.floating):
        # [-1, 1] -> [0, 255]
        img = ((img + 1.0) / 2.0 * 255).clip(0, 255).astype(np.uint8)
    Image.fromarray(img).save(path)
    logger.info("  Saved image: %s  shape=%s", path, img.shape)


def dump_observation(obs, prefix, out_dir):
    """Dump an Observation's contents to disk for inspection."""
    out_dir.mkdir(parents=True, exist_ok=True)

    # Images
    logger.info("Image key order: %s", list(obs.images.keys()))
    for i, (key, img) in enumerate(obs.images.items()):
        img_np = np.asarray(img)
        save_image(img_np, out_dir / f"{prefix}_img_{i}_{key}.png")
        logger.info("  %s: dtype=%s shape=%s min=%.3f max=%.3f",
                     key, img_np.dtype, img_np.shape, float(img_np.min()), float(img_np.max()))

    # State
    state_np = np.asarray(obs.state)
    logger.info("State: shape=%s dtype=%s", state_np.shape, state_np.dtype)
    logger.info("  values: %s", state_np.flatten()[:14])
    np.save(out_dir / f"{prefix}_state.npy", state_np)

    # Tokenized prompt
    if obs.tokenized_prompt is not None:
        tokens_np = np.asarray(obs.tokenized_prompt)
        mask_np = np.asarray(obs.tokenized_prompt_mask)
        logger.info("Tokenized prompt: shape=%s", tokens_np.shape)
        # Show non-zero tokens
        nonzero = tokens_np[tokens_np != 0] if tokens_np.ndim == 1 else tokens_np[0][tokens_np[0] != 0]
        logger.info("  Non-zero tokens (%d): %s", len(nonzero), nonzero[:50])
        logger.info("  Mask sum (valid tokens): %d", int(mask_np.sum()))
        np.save(out_dir / f"{prefix}_tokens.npy", tokens_np)
        np.save(out_dir / f"{prefix}_token_mask.npy", mask_np)

        if obs.token_loss_mask is not None:
            loss_mask_np = np.asarray(obs.token_loss_mask)
            logger.info("  Loss mask sum (subtask tokens): %d", int(loss_mask_np.sum()))
            np.save(out_dir / f"{prefix}_loss_mask.npy", loss_mask_np)

    # Image masks
    for key, mask in obs.image_masks.items():
        logger.info("  Image mask %s: %s", key, np.asarray(mask))


def run_inference_pipeline(config_name, checkpoint_path, high_prompt, low_prompt):
    """Replicate what the inference server does, without loading the model.

    This mirrors async_pi05_inference.py prepare_observation() + the server's
    normalize_state(), so we can see exactly what the model would receive.
    """
    logger.info("=" * 60)
    logger.info("INFERENCE PIPELINE")
    logger.info("=" * 60)

    os.environ.setdefault("JAX_PLATFORMS", "cpu")
    import jax
    import jax.numpy as jnp

    from openpi.models import model as _model
    from openpi.models.model import Observation
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.training.config import get_config

    config = get_config(config_name)
    model_config = config.model

    # Create tokenizer (same as AsyncPi05Inference._initialize_blocking)
    tokenizer_kwargs = {"max_len": model_config.max_token_len}
    fast_token_loss_weight = getattr(model_config, "fast_token_loss_weight", 0.0)
    if fast_token_loss_weight > 0:
        tokenizer_kwargs["fast_tokenizer_path"] = getattr(
            model_config, "fast_tokenizer_path", "physical-intelligence/fast"
        )
    tokenizer = PaligemmaTokenizer(**tokenizer_kwargs)

    # Use a synthetic 480x640 image (like a real camera frame)
    dummy_img = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    images = {
        "base_0_rgb": dummy_img.copy(),
        "left_wrist_0_rgb": dummy_img.copy(),
        "right_wrist_0_rgb": dummy_img.copy(),
        "base_1_rgb": dummy_img.copy(),
    }

    # Use a realistic state (14-dim, raw degrees)
    raw_state = np.array([3.3, -66.9, 100.0, 0.0, -43.3, -16.0, 82.2,
                          11.4, -99.5, 100.0, 2.3, 60.6, 3.3, 4.2], dtype=np.float32)

    # Load norm_stats and normalize (same as server does)
    from async_pi05_websocket_server import load_norm_stats, normalize_state
    norm_stats, norm_path = load_norm_stats(checkpoint_path, config_name)
    logger.info("Norm stats loaded from: %s", norm_path)

    normalized_state = normalize_state(raw_state, norm_stats, pad_to_dim=0, use_quantiles=True)
    logger.info("Raw state (14D): %s", raw_state)
    logger.info("Normalized state (%dD): %s", len(normalized_state), normalized_state)
    logger.info("Normalized state range: min=%.3f max=%.3f", normalized_state.min(), normalized_state.max())

    # --- Replicate prepare_observation() from async_pi05_inference.py ---
    # This is the EXACT code path used during inference.
    img_dict = {}
    image_mask_dict = {}
    for key, img in images.items():
        img_array = np.asarray(img, dtype=np.uint8)
        img_dict[key] = jnp.array(img_array[np.newaxis, :, :, :])
        image_mask_dict[key] = jnp.array(
            [not (key == "right_wrist_0_rgb" and not np.any(img_array))], dtype=jnp.bool_
        )

    # State handling — tokenize with original dim, then pad (matches training order)
    state_vec = np.asarray(normalized_state, dtype=np.float32).reshape(-1)
    logger.info("State vec dim (pre-tokenization): %d", state_vec.shape[0])

    # Tokenize with original state dim (matches training: TokenizeHighLowPrompt before PadStatesAndActions)
    (
        tokenized_prompt,
        tokenized_prompt_mask,
        token_ar_mask,
        token_loss_mask,
        _subtask_region_mask,
        _action_region_mask,
    ) = tokenizer.tokenize_high_low_prompt(high_prompt, low_prompt, state_vec)

    # Pad state to 32 AFTER tokenization (matches PadStatesAndActions)
    if state_vec.shape[0] < 32:
        state_vec = np.pad(state_vec, ((0, 32 - state_vec.shape[0])), constant_values=0.0)
    elif state_vec.shape[0] > 32:
        state_vec = state_vec[:32]
    state_batch = jnp.asarray(state_vec, dtype=jnp.float32)[np.newaxis, :]

    # Build observation
    data = {
        "image": img_dict,
        "image_mask": image_mask_dict,
        "state": state_batch,
        "tokenized_prompt": jnp.stack([tokenized_prompt], axis=0),
        "tokenized_prompt_mask": jnp.stack([tokenized_prompt_mask], axis=0),
        "token_ar_mask": jnp.stack([token_ar_mask], axis=0),
        "token_loss_mask": jnp.stack([token_loss_mask], axis=0),
    }

    observation = Observation.from_dict(data)
    rng = jax.random.key(42)
    observation = _model.preprocess_observation(
        rng, observation, train=False, image_keys=list(observation.images.keys()),
    )

    out_dir = OUT_DIR / "inference"
    dump_observation(observation, "infer", out_dir)

    # Decode tokens
    tokens_np = np.asarray(observation.tokenized_prompt)
    if tokens_np.ndim == 2:
        tokens_np = tokens_np[0]
    decoded = tokenizer.detokenize(tokens_np.astype(np.int32))
    logger.info("Decoded prompt: %s", repr(decoded))

    return observation


def run_training_pipeline(config_name, dataset_repo_id, high_prompt, low_prompt, checkpoint_path=None):
    """Load one sample from the training dataset and run through training transforms.

    Uses LeRobotDataset (v3-compatible) for data loading, then applies the
    exact same transform chain as training:
        repack → SOARMInputs → Normalize → ResizeImages → TokenizeHighLowPrompt → PadStatesAndActions
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE")
    logger.info("=" * 60)

    os.environ.setdefault("JAX_PLATFORMS", "cpu")

    from openpi.models import model as _model
    from openpi.models.model import Observation
    from openpi.policies import soarm_policy
    from openpi.training import config as _config
    from openpi.training.config import SubtaskModelTransformFactory
    from openpi import transforms as _transforms

    config = _config._CONFIGS_DICT[config_name]
    model_config = config.model

    # --- Step 1: Load one sample using LeRobotDataset (handles v3 video decoding) ---
    logger.info("Loading dataset: %s (via LeRobotDataset)", dataset_repo_id)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        dataset = LeRobotDataset(dataset_repo_id)
        sample = dataset[0]
        logger.info("Dataset sample keys: %s", list(sample.keys()))
    except Exception as e:
        logger.error("Could not load dataset: %s", e)
        return None

    # Show raw sample info
    for key in sorted(sample.keys()):
        val = sample[key]
        if hasattr(val, 'shape'):
            logger.info("  %s: shape=%s dtype=%s", key, val.shape, getattr(val, 'dtype', 'N/A'))
        else:
            logger.info("  %s: %s = %s", key, type(val).__name__, repr(val)[:80])

    # --- Step 2: Manual repack (same as LeRobotSOARMDataConfig repack_mapping) ---
    repacked = {}

    # Images: "observation.images.front" → "images.front"
    camera_mapping = {
        "observation.images.front": "images.front",
        "observation.images.left_wrist": "images.left_wrist",
        "observation.images.right_wrist": "images.right_wrist",
        "observation.images.top": "images.top",
    }
    for src_key, dst_key in camera_mapping.items():
        if src_key in sample:
            repacked[dst_key] = np.asarray(sample[src_key])
            logger.info("  Repack %s → %s: shape=%s", src_key, dst_key, repacked[dst_key].shape)

    # State and actions
    if "observation.state" in sample:
        repacked["state"] = np.asarray(sample["observation.state"], dtype=np.float32)
    if "action" in sample:
        repacked["actions"] = np.asarray(sample["action"], dtype=np.float32)

    # Task and subtask — override with our test prompts
    repacked["task"] = high_prompt
    repacked["subtask"] = low_prompt

    logger.info("Repacked keys: %s", list(repacked.keys()))

    # --- Step 3: SOARMInputs transform (converts camera names, CHW→HWC, sets prompts) ---
    soarm_transform = soarm_policy.SOARMInputs(model_type=model_config.model_type)
    transformed = soarm_transform(repacked)
    logger.info("After SOARMInputs:")
    logger.info("  Image keys: %s", list(transformed["image"].keys()))
    for key, img in transformed["image"].items():
        logger.info("    %s: shape=%s dtype=%s min=%s max=%s",
                     key, img.shape, img.dtype, img.min(), img.max())
    logger.info("  State: shape=%s dtype=%s", transformed["state"].shape, transformed["state"].dtype)
    logger.info("  State values (14D): %s", transformed["state"][:14])
    logger.info("  high_prompt: %s", transformed.get("high_prompt"))
    logger.info("  low_prompt: %s", transformed.get("low_prompt"))

    # --- Step 4: Normalize (quantile normalization, same as training) ---
    # Load norm_stats the same way training does
    data_config = config.data.create(
        pathlib.Path(config.data.assets.assets_dir or ""),
        model_config,
    )
    norm_stats = data_config.norm_stats

    if norm_stats is None:
        logger.warning("No norm_stats found via config, loading from checkpoint assets")
        from openpi.shared import normalize as _normalize
        # Search for norm_stats.json under checkpoint assets
        if checkpoint_path:
            candidates = list(pathlib.Path(checkpoint_path).rglob("norm_stats.json"))
            if candidates:
                norm_stats = _normalize.load(candidates[0].parent)
                logger.info("Loaded norm_stats from: %s", candidates[0])

    normalize_fn = _transforms.Normalize(norm_stats, use_quantiles=data_config.use_quantile_norm)
    normalized = normalize_fn(transformed)
    logger.info("After Normalize:")
    logger.info("  State values (14D): %s", normalized["state"][:14])
    logger.info("  State range: [%.3f, %.3f]",
                float(normalized["state"].min()), float(normalized["state"].max()))

    # --- Step 5: Model transforms (ResizeImages → TokenizeHighLowPrompt → PadStatesAndActions) ---
    model_transforms = SubtaskModelTransformFactory()(model_config)
    model_input = normalized
    for transform in model_transforms.inputs:
        model_input = transform(model_input)

    logger.info("After model transforms:")
    for key, img in model_input["image"].items():
        logger.info("  %s: shape=%s dtype=%s min=%.1f max=%.1f",
                     key, img.shape, img.dtype, float(img.min()), float(img.max()))
    logger.info("  State: shape=%s values=%s", model_input["state"].shape, model_input["state"][:14])
    logger.info("  State dim: %d (should be 32 after PadStatesAndActions)", model_input["state"].shape[-1])
    logger.info("  Tokens: shape=%s non-zero=%d",
                model_input["tokenized_prompt"].shape,
                int(np.count_nonzero(model_input["tokenized_prompt"])))

    # --- Step 6: Convert to Observation (same as training dataloader output) ---
    for key in model_input["image"]:
        model_input["image"][key] = model_input["image"][key][np.newaxis, ...]
    model_input["image_mask"] = model_input.get("image_mask", {})
    for key in model_input["image"]:
        mask_val = model_input["image_mask"].get(key)
        if mask_val is None or np.asarray(mask_val).ndim == 0:
            model_input["image_mask"][key] = np.array([True])
    model_input["state"] = model_input["state"][np.newaxis, ...]
    model_input["tokenized_prompt"] = model_input["tokenized_prompt"][np.newaxis, ...]
    model_input["tokenized_prompt_mask"] = model_input["tokenized_prompt_mask"][np.newaxis, ...]
    model_input["token_ar_mask"] = model_input["token_ar_mask"][np.newaxis, ...]
    model_input["token_loss_mask"] = model_input["token_loss_mask"][np.newaxis, ...]
    if "subtask_region_mask" in model_input:
        model_input["subtask_region_mask"] = model_input["subtask_region_mask"][np.newaxis, ...]
    if "action_region_mask" in model_input:
        model_input["action_region_mask"] = model_input["action_region_mask"][np.newaxis, ...]

    observation = Observation.from_dict(model_input)

    out_dir = OUT_DIR / "training"
    dump_observation(observation, "train", out_dir)

    # Decode tokens
    from openpi.models.tokenizer import PaligemmaTokenizer
    tokenizer_kwargs = {"max_len": model_config.max_token_len}
    fast_token_loss_weight = getattr(model_config, "fast_token_loss_weight", 0.0)
    if fast_token_loss_weight > 0:
        tokenizer_kwargs["fast_tokenizer_path"] = getattr(
            model_config, "fast_tokenizer_path", "physical-intelligence/fast"
        )
    tokenizer = PaligemmaTokenizer(**tokenizer_kwargs)
    tokens_np = np.asarray(observation.tokenized_prompt)
    if tokens_np.ndim == 2:
        tokens_np = tokens_np[0]
    decoded = tokenizer.detokenize(tokens_np.astype(np.int32))
    logger.info("Decoded prompt: %s", repr(decoded))

    return observation


def compare(infer_obs, train_obs):
    """Compare two observations side by side."""
    if infer_obs is None or train_obs is None:
        logger.warning("Cannot compare: one or both observations are None")
        return

    logger.info("=" * 60)
    logger.info("COMPARISON")
    logger.info("=" * 60)

    # Image key order
    infer_keys = list(infer_obs.images.keys())
    train_keys = list(train_obs.images.keys())
    match = "MATCH" if infer_keys == train_keys else "MISMATCH"
    logger.info("Image key order: %s", match)
    logger.info("  Training:  %s", train_keys)
    logger.info("  Inference: %s", infer_keys)

    # Image ranges
    for key in infer_keys:
        if key in train_obs.images:
            ti = np.asarray(train_obs.images[key])
            ii = np.asarray(infer_obs.images[key])
            logger.info("Image %s:", key)
            logger.info("  Training:  shape=%s dtype=%s range=[%.3f, %.3f]",
                         ti.shape, ti.dtype, float(ti.min()), float(ti.max()))
            logger.info("  Inference: shape=%s dtype=%s range=[%.3f, %.3f]",
                         ii.shape, ii.dtype, float(ii.min()), float(ii.max()))

    # State
    ts = np.asarray(train_obs.state).flatten()
    ist = np.asarray(infer_obs.state).flatten()
    logger.info("State:")
    logger.info("  Training dim:  %d", len(ts))
    logger.info("  Inference dim: %d", len(ist))
    logger.info("  Training  (first 14): %s", ts[:14])
    logger.info("  Inference (first 14): %s", ist[:14])
    logger.info("  Training  range: [%.3f, %.3f]", float(ts.min()), float(ts.max()))
    logger.info("  Inference range: [%.3f, %.3f]", float(ist.min()), float(ist.max()))

    # Token comparison
    if train_obs.tokenized_prompt is not None and infer_obs.tokenized_prompt is not None:
        tt = np.asarray(train_obs.tokenized_prompt).flatten()
        it = np.asarray(infer_obs.tokenized_prompt).flatten()
        tt_nonzero = tt[tt != 0]
        it_nonzero = it[it != 0]
        logger.info("Tokens:")
        logger.info("  Training non-zero:  %d tokens", len(tt_nonzero))
        logger.info("  Inference non-zero: %d tokens", len(it_nonzero))
        logger.info("  Training first 30:  %s", tt_nonzero[:30])
        logger.info("  Inference first 30: %s", it_nonzero[:30])

        # Check if first N tokens match (the task prompt portion)
        min_len = min(len(tt_nonzero), len(it_nonzero))
        if min_len > 0:
            matching = int(np.sum(tt_nonzero[:min_len] == it_nonzero[:min_len]))
            logger.info("  First %d tokens: %d match, %d differ",
                         min_len, matching, min_len - matching)

            # Find first divergence
            for i in range(min_len):
                if tt_nonzero[i] != it_nonzero[i]:
                    logger.info("  FIRST DIVERGENCE at position %d: train=%d, infer=%d",
                                i, int(tt_nonzero[i]), int(it_nonzero[i]))
                    break
            else:
                if len(tt_nonzero) != len(it_nonzero):
                    logger.info("  All %d shared tokens match, but lengths differ: train=%d, infer=%d",
                                min_len, len(tt_nonzero), len(it_nonzero))
                else:
                    logger.info("  ALL TOKENS MATCH PERFECTLY")


def main():
    parser = argparse.ArgumentParser(description="Verify inference inputs match training")
    parser.add_argument("--config", default="soarm_pi05_flow_lora", help="Training config name")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path (for inference pipeline)")
    parser.add_argument("--dataset", default="thewisp/cylinder_ring_assembly", help="HF dataset repo")
    parser.add_argument("--high-prompt", default="assemble cylinder into ring", help="High-level task")
    parser.add_argument("--low-prompt", default="pick up the cylinder", help="Low-level subtask")
    parser.add_argument("--skip-inference", action="store_true", help="Skip inference pipeline (no checkpoint needed)")
    parser.add_argument("--skip-training", action="store_true", help="Skip training pipeline (no dataset needed)")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Output directory: %s", OUT_DIR)

    infer_obs = None
    train_obs = None

    if not args.skip_training:
        train_obs = run_training_pipeline(args.config, args.dataset, args.high_prompt, args.low_prompt, args.checkpoint)

    if not args.skip_inference:
        if args.checkpoint is None:
            logger.warning("No --checkpoint specified, skipping inference pipeline")
        else:
            infer_obs = run_inference_pipeline(args.config, args.checkpoint, args.high_prompt, args.low_prompt)

    if train_obs is not None and infer_obs is not None:
        compare(infer_obs, train_obs)

    logger.info("Done. Outputs saved to %s", OUT_DIR)


if __name__ == "__main__":
    main()
