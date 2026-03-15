import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

try:
    import openpi.models.gemma as _gemma
except ImportError:
    # JAX/Flax not available — use the lightweight pure-Python config
    import openpi.models_pytorch.gemma_config_lite as _gemma  # type: ignore[no-redef]
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class PI0Pytorch(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(32, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, 32)

        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(32, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")

        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True, image_keys=None):
        """Helper method to preprocess observation."""
        kwargs = {"train": train}
        if image_keys is not None:
            kwargs["image_keys"] = image_keys
        observation = _preprocessing.preprocess_observation_pytorch(observation, **kwargs)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)

    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def embed_suffix(self, state, noisy_actions, timestep):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.state_proj.weight.dtype == torch.float32:
                state = state.to(torch.float32)

            # Embed state
            def state_proj_func(state):
                return self.state_proj(state)

            state_emb = self._apply_checkpoint(state_proj_func, state)

            embs.append(state_emb[:, None, :])
            bsize = state_emb.shape[0]
            device = state_emb.device

            state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
            pad_masks.append(state_mask)

            # Set attention masks so that image and language inputs do not attend to state or actions
            att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

            # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, time)
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)

        return F.mse_loss(u_t, v_t, reduction="none")

    @torch.no_grad()
    def extract_prefix_latent(self, device, observation, image_keys=None) -> Tensor:
        """Extract a scene-understanding latent by mean-pooling the PaliGemma prefix output.

        Returns a [B, 2048] tensor representing the S2 latent for dual-system VLA inference.
        Skips action denoising entirely — ~150ms vs ~560ms for full inference.

        Args:
            image_keys: Optional sequence of image keys to process. If None, uses
                        preprocessing default (3 cameras). Pass 4 keys for SOARM.
        """
        if image_keys is not None:
            processed = _preprocessing.preprocess_observation_pytorch(
                observation, train=False, image_keys=image_keys
            )
            images = list(processed.images.values())
            img_masks = list(processed.image_masks.values())
            lang_tokens = processed.tokenized_prompt
            lang_masks = processed.tokenized_prompt_mask
        else:
            images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_out, _), _, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=False,
        )

        # Mean-pool over valid prefix tokens → [B, 2048]
        mask = prefix_pad_masks[:, :, None].to(dtype=prefix_out.dtype)
        pooled = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        return pooled

    @torch.no_grad()
    def extract_prefix_latent_and_subtask(self, device, observation, max_decoding_steps=20, image_keys=None, temperature=0.0):
        """Single prefix forward → latent (mean-pool) + AR subtask decoding from KV cache.

        Args:
            temperature: 0.0 = greedy argmax, >0.0 = sample from softmax(logits/T).
                         Higher values explore more diverse subtasks.

        Returns (pooled_latent [B, 2048], output_tokens list[int]).
        """
        if image_keys is not None:
            processed = _preprocessing.preprocess_observation_pytorch(
                observation, train=False, image_keys=image_keys
            )
            images = list(processed.images.values())
            img_masks = list(processed.image_masks.values())
            lang_tokens = processed.tokenized_prompt
            lang_masks = processed.tokenized_prompt_mask
        else:
            images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        # Single prefix forward with cache
        (prefix_out, _), past_kv, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Extract latent via mean-pooling
        mask = prefix_pad_masks[:, :, None].to(dtype=prefix_out.dtype)
        pooled = (prefix_out * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

        # AR-decode subtask from KV cache
        def _sample_token(logits_2d):
            """Sample or argmax from [B, vocab] logits based on temperature."""
            if temperature <= 0.0:
                return logits_2d.argmax(dim=-1, keepdim=True)
            probs = torch.softmax(logits_2d / temperature, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        last_valid_idx = int(torch.where(prefix_pad_masks[0])[0][-1].item())
        logits = self.paligemma_with_expert.deembed(prefix_out[:, last_valid_idx : last_valid_idx + 1])
        next_token = _sample_token(logits[:, 0])

        prefix_valid_len = int(prefix_pad_masks[0].sum().item())
        hidden_dim = prefix_embs.shape[-1]
        emb_scale = math.sqrt(hidden_dim)

        EOS_TOKEN = 1
        output_tokens = []
        # Collect top-k info at each AR step for debugging
        topk_per_step = []  # list of [(token_id, prob), ...] per step

        # Log top-k for the first token from prefix (raw logits, not softmax)
        first_topk = torch.topk(logits[0, 0], k=min(5, logits.shape[-1]))
        topk_per_step.append(list(zip(first_topk.indices.tolist(), first_topk.values.tolist())))

        for step in range(max_decoding_steps):
            tok = next_token[0, 0].item()
            if tok == EOS_TOKEN:
                break
            output_tokens.append(tok)

            token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)
            token_emb = token_emb * emb_scale
            token_emb = token_emb.to(dtype=prefix_embs.dtype)

            pos_ids = torch.tensor([[prefix_valid_len + step]], device=device, dtype=torch.long)

            NEG_INF = -2.3819763e38
            cross = torch.where(prefix_pad_masks[0], 0.0, NEG_INF)
            ar_part = torch.zeros(step + 1, device=device, dtype=torch.float32)
            ar_att_4d = torch.cat([cross, ar_part], dim=0).reshape(1, 1, 1, -1)

            (ar_out, _), past_kv, _ = self.paligemma_with_expert.forward(
                attention_mask=ar_att_4d,
                position_ids=pos_ids,
                past_key_values=past_kv,
                inputs_embeds=[token_emb, None],
                use_cache=True,
            )

            logits = self.paligemma_with_expert.deembed(ar_out[:, -1:])
            # Collect top-k raw logits before sampling
            step_topk = torch.topk(logits[0, 0], k=min(5, logits.shape[-1]))
            topk_per_step.append(list(zip(step_topk.indices.tolist(), step_topk.values.tolist())))

            next_token = _sample_token(logits[:, 0])

        return pooled, output_tokens, topk_per_step

    @torch.no_grad()
    def sample_low_level_task(self, device, observation, max_decoding_steps=20, image_keys=None, return_cache=False):
        """AR-decode a subtask description from the PaliGemma prefix LM.

        Mirrors the JAX `sample_low_level_task` logic: runs the prefix forward with
        KV-cache enabled, then greedily decodes tokens one-by-one until EOS or
        `max_decoding_steps` is reached.

        Returns a list of integer token IDs (not including EOS).
        """
        if image_keys is not None:
            processed = _preprocessing.preprocess_observation_pytorch(
                observation, train=False, image_keys=image_keys
            )
            images = list(processed.images.values())
            img_masks = list(processed.image_masks.values())
            lang_tokens = processed.tokenized_prompt
            lang_masks = processed.tokenized_prompt_mask
        else:
            images, img_masks, lang_tokens, lang_masks, _ = self._preprocess_observation(observation, train=False)

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        (prefix_out, _), past_kv, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # Logits from the last VALID prefix token → first generated token
        # (last position may be padding; use the last True index in pad_masks)
        last_valid_idx = int(torch.where(prefix_pad_masks[0])[0][-1].item())
        logits = self.paligemma_with_expert.deembed(prefix_out[:, last_valid_idx : last_valid_idx + 1])  # [B, 1, vocab]
        next_token = logits[:, 0].argmax(dim=-1, keepdim=True)  # [B, 1]

        prefix_len = prefix_embs.shape[1]
        prefix_valid_len = int(prefix_pad_masks[0].sum().item())
        hidden_dim = prefix_embs.shape[-1]
        emb_scale = math.sqrt(hidden_dim)

        EOS_TOKEN = 1
        output_tokens = []

        for step in range(max_decoding_steps):
            tok = next_token[0, 0].item()
            if tok == EOS_TOKEN:
                break
            output_tokens.append(tok)

            # Embed the generated token (with same scale as embed_prefix)
            token_emb = self.paligemma_with_expert.embed_language_tokens(next_token)  # [B, 1, D]
            token_emb = token_emb * emb_scale
            token_emb = token_emb.to(dtype=prefix_embs.dtype)

            # Position: right after all valid prefix tokens
            pos_ids = torch.tensor([[prefix_valid_len + step]], device=device, dtype=torch.long)

            # Attention mask: current AR token attends to valid prefix positions + all AR tokens.
            # Match denoise_step approach: mask out padded prefix positions (garbage KV values).
            NEG_INF = -2.3819763e38
            cross = torch.where(prefix_pad_masks[0], 0.0, NEG_INF)  # [prefix_len], float32
            ar_part = torch.zeros(step + 1, device=device, dtype=torch.float32)
            ar_att_4d = torch.cat([cross, ar_part], dim=0).reshape(1, 1, 1, -1)

            (ar_out, _), past_kv, _ = self.paligemma_with_expert.forward(
                attention_mask=ar_att_4d,
                position_ids=pos_ids,
                past_key_values=past_kv,
                inputs_embeds=[token_emb, None],
                use_cache=True,
            )

            logits = self.paligemma_with_expert.deembed(ar_out[:, -1:])  # [B, 1, vocab]
            next_token = logits[:, 0].argmax(dim=-1, keepdim=True)

        if not return_cache:
            return output_tokens

        # Extend prefix_pad_masks to cover the AR-decoded tokens (all valid).
        # The KV cache now has length prefix_len + len(output_tokens); the caller
        # needs correct pad masks to let the action expert cross-attend to them.
        n_decoded = len(output_tokens)
        if n_decoded > 0:
            extra = torch.ones(prefix_pad_masks.shape[0], n_decoded, dtype=torch.bool, device=device)
            extended_pad_masks = torch.cat([prefix_pad_masks, extra], dim=1)
        else:
            extended_pad_masks = prefix_pad_masks
        return output_tokens, past_kv, extended_pad_masks

    @torch.no_grad()
    def sample_actions_from_cache(self, device, past_key_values, prefix_pad_masks, state, noise=None, num_steps=10):
        """Flow-matching action generation using a pre-computed prefix KV cache.

        Skips the prefix forward entirely — use the KV cache (and extended
        prefix_pad_masks) returned by sample_low_level_task(return_cache=True).
        Saves ~50ms per inference vs calling sample_actions separately.
        """
        bsize = state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            v_t = self.denoise_step(state, prefix_pad_masks, past_key_values, x_t, time.expand(bsize))
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    @torch.no_grad()
    def sample_actions(self, device, observation, noise=None, num_steps=10, image_keys=None) -> Tensor:
        """Do a full inference forward and compute the action (batch_size x num_steps x num_motors)"""
        bsize = observation.state.shape[0]
        if noise is None:
            actions_shape = (bsize, self.config.action_horizon, self.config.action_dim)
            noise = self.sample_noise(actions_shape, device)

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(
            observation, train=False, image_keys=image_keys
        )

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # Compute image and language key value cache
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"  # noqa: SLF001

        _, past_key_values, _ = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        dt = -1.0 / num_steps
        dt = torch.tensor(dt, dtype=torch.float32, device=device)

        x_t = noise
        time = torch.tensor(1.0, dtype=torch.float32, device=device)
        while time >= -dt / 2:
            expanded_time = time.expand(bsize)
            v_t = self.denoise_step(
                state,
                prefix_pad_masks,
                past_key_values,
                x_t,
                expanded_time,
            )

            # Euler step - use new tensor assignment instead of in-place operation
            x_t = x_t + dt * v_t
            time += dt
        return x_t

    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)
