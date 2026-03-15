"""
PyTorch inference engine for Pi0.5 — drop-in replacement for the JAX server.

Loads the converted safetensors checkpoint and runs full inference:
  - extract_latent(): prefix encoding only → [2048] S2 latent (~50ms warm)
  - infer(): AR subtask decoding + action generation (~230ms warm)

15× faster than the JAX server (840ms → ~60ms for actions, ~230ms with subtask).
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

_PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(_PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

# SOARM uses 4 cameras — must match training configuration
SOARM_IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb")


def resolve_checkpoint_path(checkpoint: str) -> str:
    """Accept either a directory (containing model.safetensors) or a direct file path."""
    p = Path(checkpoint)
    if p.is_dir():
        candidate = p / "model.safetensors"
        if not candidate.exists():
            raise FileNotFoundError(f"No model.safetensors found in {p}")
        return str(candidate)
    return str(p)


class SimpleObservation:
    """Minimal observation container for PyTorch preprocessing."""
    def __init__(self, images, image_masks, state, tokenized_prompt, tokenized_prompt_mask,
                 token_ar_mask, token_loss_mask):
        self.images = images
        self.image_masks = image_masks
        self.state = state
        self.tokenized_prompt = tokenized_prompt
        self.tokenized_prompt_mask = tokenized_prompt_mask
        self.token_ar_mask = token_ar_mask
        self.token_loss_mask = token_loss_mask


class PyTorchPi05Inference:
    """PyTorch-based Pi0.5 inference engine.

    Drop-in replacement for AsyncPi05Inference with ~15× speedup over JAX.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = "cuda:0",
        image_keys: tuple[str, ...] = SOARM_IMAGE_KEYS,
        norm_stats_path: str | None = None,
    ):
        self.checkpoint_path = resolve_checkpoint_path(checkpoint_path)
        self.device = torch.device(device)
        self.image_keys = image_keys
        self.norm_stats = self._load_norm_stats(norm_stats_path)

        self.model = None
        self.tokenizer = None
        self._initialized = False
        self._initialize_lock = asyncio.Lock()
        self._model_lock = asyncio.Lock()

    @staticmethod
    def _load_norm_stats(path: str | None) -> dict | None:
        if path is None:
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            ns = data.get("norm_stats", data)
            logger.info("Loaded norm_stats from %s (state dim=%d)", path, len(ns["state"]["q01"]))
            return ns
        except Exception as e:
            logger.warning("Could not load norm_stats from %s: %s — state will not be normalized", path, e)
            return None

    def _normalize_state(self, state: np.ndarray) -> np.ndarray:
        """Quantile-normalize state to [-1, 1], matching the JAX server."""
        if self.norm_stats is None or "state" not in self.norm_stats:
            return state
        stats = self.norm_stats["state"]
        q01 = np.array(stats["q01"], dtype=np.float32)
        q99 = np.array(stats["q99"], dtype=np.float32)
        dim = min(state.shape[-1], len(q01))
        normalized = state.copy()
        normalized[..., :dim] = (
            (state[..., :dim] - q01[:dim]) / (q99[:dim] - q01[:dim] + 1e-6) * 2.0 - 1.0
        )
        return normalized

    def _unnormalize_actions(self, actions: np.ndarray) -> np.ndarray:
        """Inverse quantile normalization for actions, matching JAX server."""
        if self.norm_stats is None or "actions" not in self.norm_stats:
            return actions
        stats = self.norm_stats["actions"]
        q01 = np.array(stats["q01"], dtype=np.float32)
        q99 = np.array(stats["q99"], dtype=np.float32)
        dim = min(actions.shape[-1], len(q01))
        out = actions.copy()
        out[..., :dim] = (
            (actions[..., :dim] + 1.0) / 2.0 * (q99[:dim] - q01[:dim] + 1e-6) + q01[:dim]
        )
        return out

    async def _run_blocking(self, fn, *, use_model_lock: bool = False):
        loop = asyncio.get_running_loop()
        if use_model_lock:
            async with self._model_lock:
                return await loop.run_in_executor(None, fn)
        return await loop.run_in_executor(None, fn)

    def _initialize_blocking(self):
        import safetensors.torch as st
        from openpi.models_pytorch.pi0_pytorch import PI0Pytorch
        from openpi.models_pytorch.gemma_config_lite import Pi0ConfigLite

        logger.info("Loading PyTorch Pi0.5 from %s", self.checkpoint_path)

        model_config = Pi0ConfigLite(
            action_horizon=50,
            action_dim=32,
            paligemma_variant="gemma_2b",
            pi05=True,
        )
        model = PI0Pytorch(model_config)

        logger.info("Loading safetensors checkpoint...")
        sd = st.load_file(self.checkpoint_path)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        if missing:
            logger.warning("Missing keys: %s", missing[:5])
        if unexpected:
            logger.warning("Unexpected keys: %s", unexpected[:5])

        self.model = model.to(self.device).eval()

        # Compile the PaliGemma language model forward for faster AR decode.
        # dynamic=True handles the growing KV cache and changing attention mask shapes.
        # reduce-overhead mode minimizes per-step CUDA kernel launch overhead.
        # Note: first inference after warmup will still be slow (recompilation on new shapes).
        try:
            self.model.paligemma_with_expert.forward = torch.compile(
                self.model.paligemma_with_expert.forward,
                dynamic=True,
                mode="reduce-overhead",
            )
            logger.info("torch.compile applied to paligemma_with_expert.forward")
        except Exception as e:
            logger.warning("torch.compile failed, falling back to eager: %s", e)

        try:
            from openpi.models.tokenizer import PaligemmaTokenizer
            self.tokenizer = PaligemmaTokenizer(max_len=256)
        except Exception:
            from openpi.models.tokenizer_lite import PaligemmaTokenizerLite
            self.tokenizer = PaligemmaTokenizerLite(max_len=256)

        logger.info("PyTorch Pi0.5 loaded on %s", self.device)

    def _warmup_blocking(self):
        """Run one dummy forward pass to trigger Triton autotuning before serving requests."""
        logger.info("Running warm-up inference to compile CUDA kernels (may take 2-3 min)...")
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        dummy_images = {k: dummy_img for k in self.image_keys}
        obs = self._prepare_observation(dummy_images, "warm up", "", None)
        with torch.no_grad():
            self.model.sample_actions(self.device, obs, num_steps=10, image_keys=self.image_keys)
        logger.info("Warm-up complete. Server ready for fast inference.")

    async def initialize(self, warmup: bool = True):
        if self._initialized:
            return
        async with self._initialize_lock:
            if self._initialized:
                return
            await self._run_blocking(self._initialize_blocking, use_model_lock=True)
            self._initialized = True
            if warmup:
                await self._run_blocking(self._warmup_blocking, use_model_lock=True)

    def _prepare_observation(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        low_level_prompt: str = "",
        state: np.ndarray | None = None,
        subtask_token_ids: list[int] | None = None,
    ) -> SimpleObservation:
        """Convert raw images + text into a SimpleObservation for PyTorch preprocessing.

        If subtask_token_ids is provided, those raw token IDs (e.g. from AR decoding,
        including FAST tokens) are spliced directly into the subtask region instead of
        re-encoding low_level_prompt as text. This preserves FAST token conditioning.
        """
        if state is None:
            state_vec = np.zeros((14,), dtype=np.float32)
        else:
            state_vec = np.asarray(state, dtype=np.float32).reshape(-1)
            state_vec = self._normalize_state(state_vec)

        (
            tokenized_prompt,
            tokenized_prompt_mask,
            token_ar_mask,
            token_loss_mask,
            subtask_region_mask,
            _action_region_mask,
        ) = self.tokenizer.tokenize_high_low_prompt(high_level_prompt, low_level_prompt, state_vec)

        if subtask_token_ids is not None:
            # Splice raw decoded token IDs (including FAST tokens) directly into the
            # subtask region, bypassing text re-encoding that would discard FAST tokens.
            # subtask_region_mask marks where the empty subtask placeholder was placed;
            # we replace those positions with our decoded tokens + suffix + EOS.
            EOS = 1
            subtask_region_np = np.asarray(subtask_region_mask, dtype=bool)
            prefix_len = int(np.argmax(subtask_region_np))  # index where subtask region starts

            # subtask_token_ids already contain ";\nAction: " + FAST tokens + "|"
            # (AR decode stops before EOS). Just re-append EOS to close the sequence.
            new_subtask_seq = list(subtask_token_ids) + [EOS]

            max_len = len(tokenized_prompt)
            available = max_len - prefix_len
            if len(new_subtask_seq) > available:
                logger.warning(
                    "Subtask+suffix length %d exceeds available %d positions, truncating",
                    len(new_subtask_seq), available,
                )
                new_subtask_seq = new_subtask_seq[:available]

            n_new = len(new_subtask_seq)
            tokenized_prompt = list(tokenized_prompt)
            tokenized_prompt_mask = list(tokenized_prompt_mask)
            token_ar_mask = list(token_ar_mask)
            token_loss_mask = list(token_loss_mask)

            # Fill subtask region with decoded tokens, then pad remainder
            for i, tok in enumerate(new_subtask_seq):
                tokenized_prompt[prefix_len + i] = tok
                tokenized_prompt_mask[prefix_len + i] = True
                token_ar_mask[prefix_len + i] = 1
                token_loss_mask[prefix_len + i] = True
            # Zero out any remaining positions after the new subtask sequence
            for i in range(n_new, max_len - prefix_len):
                tokenized_prompt[prefix_len + i] = 0
                tokenized_prompt_mask[prefix_len + i] = False
                token_ar_mask[prefix_len + i] = 0
                token_loss_mask[prefix_len + i] = False

            tokenized_prompt = np.asarray(tokenized_prompt, dtype=np.int32)
            tokenized_prompt_mask = np.asarray(tokenized_prompt_mask, dtype=bool)
            token_ar_mask = np.asarray(token_ar_mask, dtype=np.int32)
            token_loss_mask = np.asarray(token_loss_mask, dtype=bool)

        # Pad/clip state to action_dim=32
        if state_vec.shape[0] < 32:
            state_vec = np.pad(state_vec, ((0, 32 - state_vec.shape[0])))
        elif state_vec.shape[0] > 32:
            state_vec = state_vec[:32]

        dev = self.device
        img_tensors = {}
        img_masks = {}
        for key in self.image_keys:
            if key in images:
                arr = np.asarray(images[key], dtype=np.uint8)
                if arr.ndim == 3:
                    arr = arr[np.newaxis]  # [1, H, W, C]
                t = torch.from_numpy(arr).to(dev).float() / 127.5 - 1.0  # normalize to [-1, 1]
                t = t.permute(0, 3, 1, 2)  # [1, H, W, C] → [1, C, H, W]
                img_tensors[key] = t
                img_masks[key] = torch.ones(1, dtype=torch.bool, device=dev)
            else:
                # Camera not available — zero image, masked out
                img_tensors[key] = torch.zeros(1, 224, 224, 3, dtype=torch.float32, device=dev)
                img_masks[key] = torch.zeros(1, dtype=torch.bool, device=dev)

        state_t = torch.from_numpy(state_vec).to(dev).unsqueeze(0)  # [1, 32]
        prompt_t = torch.from_numpy(np.array(tokenized_prompt)).to(dev).unsqueeze(0).long()
        prompt_mask_t = torch.from_numpy(np.array(tokenized_prompt_mask)).to(dev).unsqueeze(0).bool()
        ar_mask_t = torch.from_numpy(np.array(token_ar_mask)).to(dev).unsqueeze(0).long()
        loss_mask_t = torch.from_numpy(np.array(token_loss_mask)).to(dev).unsqueeze(0).bool()

        if subtask_token_ids is None:
            # For subtask decoding pass: zero out subtask region so prefix ends at "Subtask: "
            loss_mask_np = np.array(token_loss_mask) if not isinstance(token_loss_mask, np.ndarray) else token_loss_mask
            prompt_t[0, loss_mask_np] = 0
            prompt_mask_t[0, loss_mask_np] = False

        return SimpleObservation(
            images=img_tensors,
            image_masks=img_masks,
            state=state_t,
            tokenized_prompt=prompt_t,
            tokenized_prompt_mask=prompt_mask_t,
            token_ar_mask=ar_mask_t,
            token_loss_mask=loss_mask_t,
        )

    async def extract_latent(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        state: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Extract S2 prefix latent from Pi0.5 without generating actions.

        Returns dict with 's2_latent' ([2048] float32 numpy array) and 'timing'.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        obs = await self._run_blocking(
            lambda: self._prepare_observation(images, high_level_prompt, "", state),
        )

        def _extract():
            return self.model.extract_prefix_latent(self.device, obs, image_keys=self.image_keys)

        latent = await self._run_blocking(_extract, use_model_lock=True)
        latent_np = latent[0].float().cpu().numpy()  # [2048]

        total_ms = (time.time() - start_time) * 1000
        logger.debug("Prefix latent extraction: %.1fms", total_ms)
        return {
            "s2_latent": latent_np,
            "timing": {"total_ms": total_ms, "prefix_ms": total_ms},
        }

    async def extract_latent_with_subtask(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        state: np.ndarray | None = None,
        max_ar_steps: int = 20,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Extract S2 prefix latent AND AR-decode subtask in one prefix forward pass.

        Runs prefix with use_cache=True, extracts latent via mean-pooling,
        then AR-decodes subtask tokens using the cached KV. Only one prefix
        forward (no double computation).
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()
        obs = await self._run_blocking(
            lambda: self._prepare_observation(images, high_level_prompt, "", state),
        )

        _temp = temperature  # capture for closure

        def _extract_and_decode():
            t0 = time.time()
            import torch as _torch

            with _torch.no_grad():
                latent, subtask_tokens, topk_per_step = self.model.extract_prefix_latent_and_subtask(
                    self.device, obs, max_decoding_steps=max_ar_steps,
                    image_keys=self.image_keys, temperature=_temp,
                )
                latent_np = latent[0].float().cpu().numpy()
                total = (time.time() - t0) * 1000

            return latent_np, subtask_tokens, topk_per_step, total

        latent_np, subtask_tokens, topk_per_step, fused_ms = await self._run_blocking(
            _extract_and_decode, use_model_lock=True
        )

        subtask_text = ""
        if subtask_tokens:
            try:
                subtask_text = self.tokenizer.detokenize(np.array(subtask_tokens))
                if ";" in subtask_text:
                    subtask_text = subtask_text.split(";")[0].strip()
            except Exception as e:
                logger.warning("Failed to detokenize subtask tokens %s: %s", subtask_tokens, e)

        # Log top-k alternatives at each AR step (DEBUG only — very verbose)
        if topk_per_step and logger.isEnabledFor(logging.DEBUG):
            for step_i, step_topk in enumerate(topk_per_step):
                top3 = step_topk[:3]
                parts = []
                for tok_id, logit_val in top3:
                    try:
                        tok_text = self.tokenizer.detokenize(np.array([tok_id]))
                    except Exception:
                        tok_text = f"<{tok_id}>"
                    parts.append(f"{tok_text!r}({tok_id}) logit={logit_val:.1f}")
                logger.debug("  AR step %d top3: %s", step_i, " | ".join(parts))
                if step_i >= 8:
                    break

        total_ms = (time.time() - start_time) * 1000
        # Periodic INFO summary (every 10s), detailed timing at DEBUG
        now = time.monotonic()
        if not hasattr(self, '_last_subtask_log') or now - self._last_subtask_log >= 10.0:
            self._last_subtask_log = now
            logger.info("Latent+subtask: fused=%.1fms total=%.1fms subtask=%r", fused_ms, total_ms, subtask_text)
        else:
            logger.debug("Latent+subtask: fused=%.1fms total=%.1fms subtask=%r", fused_ms, total_ms, subtask_text)
        return {
            "s2_latent": latent_np,
            "subtask": subtask_text,
            "timing": {"total_ms": total_ms, "prefix_ms": fused_ms},
        }

    async def infer(
        self,
        images: dict[str, np.ndarray],
        high_level_prompt: str,
        state: np.ndarray | None = None,
        num_steps: int = 10,
        max_ar_steps: int = 20,
    ) -> dict[str, Any]:
        """Full Pi0.5 inference: AR-decode subtask, then generate action chunk.

        Matches JAX sample_actions() behavior: single prefix forward with KV cache
        reuse. sample_low_level_task(return_cache=True) produces a KV cache
        containing prefix + AR-decoded subtask tokens, then
        sample_actions_from_cache() runs flow-matching denoising using that cache.
        No redundant second prefix forward pass.

        Returns dict with 'actions', 'subtask' (str), and 'timing'.
        """
        if not self._initialized:
            await self.initialize()

        start_time = time.time()

        obs = await self._run_blocking(
            lambda: self._prepare_observation(images, high_level_prompt, "", state),
        )

        def _decode_and_act():
            t0 = time.time()
            with torch.no_grad():
                # Single prefix forward + AR subtask decode, keep KV cache
                subtask_toks, past_kv, ext_pad_masks = self.model.sample_low_level_task(
                    self.device, obs, max_decoding_steps=max_ar_steps,
                    image_keys=self.image_keys, return_cache=True,
                )
            subtask_ms = (time.time() - t0) * 1000

            t1 = time.time()
            with torch.no_grad():
                # Reuse KV cache for action denoising (matches JAX sample_actions)
                actions = self.model.sample_actions_from_cache(
                    self.device, past_kv, ext_pad_masks,
                    obs.state, num_steps=num_steps,
                )
            action_ms = (time.time() - t1) * 1000

            return subtask_toks, actions[0].float().cpu().numpy(), subtask_ms, action_ms

        subtask_tokens, actions_np, subtask_ms, action_ms = await self._run_blocking(
            _decode_and_act, use_model_lock=True
        )
        actions_np = self._unnormalize_actions(actions_np)

        subtask_text = ""
        if subtask_tokens:
            try:
                subtask_text = self.tokenizer.detokenize(np.array(subtask_tokens))
                if ";" in subtask_text:
                    subtask_text = subtask_text.split(";")[0].strip()
                logger.debug("Decoded subtask: %r  raw_tokens=%s", subtask_text, subtask_tokens)
            except Exception as e:
                logger.warning("Failed to detokenize subtask tokens %s: %s", subtask_tokens, e)

        total_ms = (time.time() - start_time) * 1000
        now = time.monotonic()
        if not hasattr(self, '_last_infer_log') or now - self._last_infer_log >= 10.0:
            self._last_infer_log = now
            logger.info("Full infer: subtask=%.1fms action=%.1fms total=%.1fms subtask=%r",
                        subtask_ms, action_ms, total_ms, subtask_text)
        else:
            logger.debug("Full infer: subtask=%.1fms action=%.1fms total=%.1fms subtask=%r",
                         subtask_ms, action_ms, total_ms, subtask_text)
        return {
            "actions": actions_np.tolist(),
            "subtask": subtask_text,
            "timing": {"total_ms": total_ms, "subtask_ms": subtask_ms, "action_ms": action_ms},
        }
