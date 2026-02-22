"""Policy transforms for SO-ARM100/107 dual-arm robots with LeRobot datasets.

Robot-specific (SO107 dual-arm, 14 DoF), not task-specific.
Handles both subtask and non-subtask datasets in one file.

Camera mapping:
    observation.images.front       -> base_0_rgb       (static front camera)
    observation.images.left_wrist  -> left_wrist_0_rgb  (left wrist cam)
    observation.images.right_wrist -> right_wrist_0_rgb (right wrist cam)
    observation.images.top         -> base_1_rgb        (static top camera)
"""

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

SOARM_STATE_DIM = 14
SOARM_ACTION_DIM = 14


def make_soarm_example() -> dict:
    """Creates a random input example for testing the SOARM policy."""
    return {
        "images.front": np.random.randint(256, size=(3, 128, 128), dtype=np.uint8),
        "images.left_wrist": np.random.randint(256, size=(3, 128, 128), dtype=np.uint8),
        "images.right_wrist": np.random.randint(256, size=(3, 128, 128), dtype=np.uint8),
        "images.top": np.random.randint(256, size=(3, 128, 128), dtype=np.uint8),
        "state": np.random.rand(SOARM_STATE_DIM).astype(np.float32),
        "actions": np.random.rand(50, SOARM_ACTION_DIM).astype(np.float32),
        "task": "assemble cylinder into socket",
        "subtask": "pick up the cylinder",
    }


def _parse_image(image: np.ndarray) -> np.ndarray:
    """Parse LeRobot image (CHW uint8 or float) to model input format (HWC uint8)."""
    image = np.asarray(image)

    if np.issubdtype(image.dtype, np.floating):
        image = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)

    if image.ndim == 3 and image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")

    return image


def _get_prompt(data: dict) -> str | None:
    """Extract task prompt from data, checking multiple possible key names."""
    for key in ("prompt", "task", "high_level_prompt"):
        if key in data:
            prompt = data[key]
            return prompt.decode("utf-8") if isinstance(prompt, bytes) else prompt
    return None


def _get_subtask(data: dict) -> str | None:
    """Extract subtask prompt from data if present."""
    if "subtask" in data:
        subtask = data["subtask"]
        return subtask.decode("utf-8") if isinstance(subtask, bytes) else subtask
    return None


@dataclasses.dataclass(frozen=True)
class SOARMInputs(transforms.DataTransformFn):
    """Transforms SOARM data dict into OpenPI model input format.

    Handles both subtask and non-subtask cases:
    - If "subtask" is present: sets high_prompt (task) and low_prompt (subtask)
    - If "subtask" is absent: sets prompt (task only)
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # 1. State: 14-dim (7 per arm: 6 joints + 1 gripper)
        state = np.asarray(data["state"], dtype=np.float32)
        if state.shape != (SOARM_STATE_DIM,):
            raise ValueError(f"SOARM state dimension error, expected {SOARM_STATE_DIM}, got {state.shape}")

        # 2. Images: 4 cameras
        front_image = _parse_image(data["images.front"])
        left_wrist_image = _parse_image(data["images.left_wrist"])
        right_wrist_image = _parse_image(data["images.right_wrist"])
        top_image = _parse_image(data["images.top"])

        # 3. Organize images for model
        match self.model_type:
            case _model.ModelType.PI0 | _model.ModelType.PI05:
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb")
                images = (front_image, left_wrist_image, right_wrist_image, top_image)
                image_masks = (np.True_, np.True_, np.True_, np.True_)
            case _model.ModelType.PI0_FAST:
                image_names = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "base_1_rgb")
                images = (front_image, left_wrist_image, right_wrist_image, top_image)
                image_masks = (np.True_, np.True_, np.True_, np.True_)
            case _:
                raise ValueError(f"Unsupported model type: {self.model_type}")

        # 4. Build input dict
        inputs = {
            "state": state,
            "image": dict(zip(image_names, images, strict=True)),
            "image_mask": dict(zip(image_names, image_masks, strict=True)),
        }

        # 5. Actions (training only)
        if "actions" in data:
            actions = np.asarray(data["actions"], dtype=np.float32)
            if actions.shape[-1] != SOARM_ACTION_DIM:
                raise ValueError(
                    f"SOARM actions dimension error, expected {SOARM_ACTION_DIM}, got {actions.shape[-1]}"
                )
            inputs["actions"] = actions

        # 6. Prompt handling — subtask-aware
        subtask = _get_subtask(data)
        if subtask is not None:
            # Subtask mode: set high_prompt and low_prompt for SubtaskModelTransformFactory
            prompt = _get_prompt(data)
            if prompt is not None:
                inputs["high_prompt"] = prompt
            inputs["low_prompt"] = subtask
        else:
            # Non-subtask mode: set prompt for ModelTransformFactory
            prompt = _get_prompt(data)
            if prompt is not None:
                inputs["prompt"] = prompt

        return inputs


@dataclasses.dataclass(frozen=True)
class SOARMOutputs(transforms.DataTransformFn):
    """Extracts the first 14 action dims from the padded model output."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"])[:, :SOARM_ACTION_DIM]}
