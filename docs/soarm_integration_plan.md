# Integration Plan: LeRobot + OpenPI Subtask Fork

## Context

You have two LeRobot-format datasets (dual SO107, 14 DoF, 4 cameras) and want to use the OpenPI subtask fork for training and inference. By training first with OpenPI, we produce a JAX checkpoint that the existing async server can serve directly — no PyTorch↔JAX conversion needed.

---

## Concepts: What Each Piece Does

| Concept | What it is | Scope | Example |
|---|---|---|---|
| **Policy transforms** (`soarm_policy.py`) | Maps key names between LeRobot format and OpenPI model format. Camera keys, state/action padding, image format. | **Robot-specific** (SO107) — shared across ALL tasks | `observation.images.front` → `base_0_rgb`, 14-dim state → padded to 32 |
| **Data config** (`LeRobotSOARMDataConfig`) | Factory that wires policy transforms + norm stats. Default `repo_id` overridable from CLI. | One per robot type, dataset swappable via CLI | `--data.repo-id=thewisp/pickup_socket_merged_head` |
| **Training config** (`TrainConfig`) | References data config + model config + hyperparameters. All overridable from CLI via tyro. | One base config, variations via CLI flags | `--batch-size=16 --num-train-steps=40000` |

---

## Milestone 1: Train from LeRobot Datasets Using OpenPI

### Steps

#### 1.1 Clone forked repo and add this plan
```bash
cd /home/feit/Documents
git clone https://github.com/TheWisp/openpi_subtask.git
cd openpi_subtask
git remote add upstream https://github.com/Ke-Wang1017/openpi_subtask.git
# Follow their install instructions (uv-based)
```

Copy this plan into the repo as `docs/soarm_integration_plan.md` so it lives alongside the code.

#### 1.2 Create SOARM policy transforms

**Create**: `src/openpi/policies/soarm_policy.py`

Reference: `src/openpi/policies/libero_subtask_policy.py`

This file is **robot-specific** (SO107 dual-arm), not task-specific. It handles both subtask and non-subtask cases in one file.

Defines `SOARMInputs` / `SOARMOutputs`:

| LeRobot Key | OpenPI Key | Notes |
|---|---|---|
| `observation.images.front` | `base_0_rgb` | Static front camera |
| `observation.images.left_wrist` | `left_wrist_0_rgb` | Left wrist cam |
| `observation.images.right_wrist` | `right_wrist_0_rgb` | Right wrist cam |
| `observation.images.top` | `base_1_rgb` | Static top cam |
| `observation.state` | `state` | 14-dim → padded to 32 |
| `action` | `actions` | 14-dim → padded to 32 |
| task description | `prompt` | High-level task |
| subtask description | `subtask` | Optional — used when present in dataset |

`SOARMInputs` handles `subtask` gracefully: if `"subtask"` string is present in the data dict, it sets `low_prompt` (subtask) and `high_prompt` (overall_task); if absent, it uses just the task prompt. This way one policy file covers both training modes. Follows the same pattern as `LiberoSubtaskInputs`.

#### 1.3 Create data config

**Modify**: `src/openpi/training/config.py`

Add `LeRobotSOARMDataConfig` (modeled on `LeRobotLiberoSubtaskDataConfig`):
- Default `repo_id` — overridable from CLI: `--data.repo-id=thewisp/other_dataset`
- `repack_transforms`: maps LeRobot keys → OpenPI internal keys
- `data_transforms`: `SOARMInputs` (camera key mapping, image format conversion)
- `use_quantile_norm`: True (matches LeRobot's quantile normalization)

No hardcoded paths — dataset is swapped at the command line.

#### 1.4 Create training config

**Modify**: `src/openpi/training/config.py`

Add ONE base config `soarm_pi05_flow`. OpenPI uses **tyro** (`overridable_config_cli`), so every field is a CLI flag.

```python
TrainConfig(
    name="soarm_pi05_flow",
    model=Pi05Config(
        action_horizon=50,
        max_token_len=200,
        subtask_loss_weight=0.0,       # Toggle for subtask training
        fast_token_loss_weight=0.0,
        flow_matching_loss_weight=1.0,
    ),
    weight_loader=CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi05_base/params"),
    data=LeRobotSOARMDataConfig(),
    batch_size=8,
    num_train_steps=20000,
    fsdp_devices=1,
)
```

Override anything at runtime:
```bash
# Different dataset
uv run scripts/train.py soarm_pi05_flow --data.repo-id=thewisp/pickup_socket_merged_head

# Enable subtask training (when dataset has subtask annotations)
uv run scripts/train.py soarm_pi05_flow --model.subtask-loss-weight=10.0 --model.fast-token-loss-weight=1.0

# Different batch size, more steps
uv run scripts/train.py soarm_pi05_flow --batch-size=16 --num-train-steps=40000
```

#### 1.5 Compute normalization stats
```bash
uv run scripts/compute_norm_stats.py --config-name soarm_pi05_flow
```

#### 1.6 Train
```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 uv run scripts/train.py soarm_pi05_flow \
  --exp-name=soarm_v1 --overwrite
```

#### 1.7 Verification
1. Norm stats computation succeeds and values match LeRobot's stored stats
2. Training smoke test (10 steps) — loss computes without errors
3. Full training — loss curve decreases on wandb

---

## Milestone 2: Serve Trained Checkpoint via OpenPI Async Server

Since training produces a JAX checkpoint, we use the **existing** OpenPI async server directly — no conversion or new server needed.

### Steps

#### 2.1 Start server
```bash
python scripts/async_pi05/async_pi05_websocket_server.py \
  --config soarm_pi05_flow \
  --checkpoint ./checkpoints/soarm_pi05_flow/soarm_v1/20000 \
  --port 8765
```

#### 2.2 Create robot client

**Create**: A thin client in LeRobot (or standalone) that:
1. Captures observations from robot (cameras + joint state)
2. Packs them as msgpack with OpenPI's expected keys (`base_0_rgb`, `left_wrist_0_rgb`, etc.)
3. Sends via websocket, receives action chunk
4. Executes first action on robot

#### 2.3 Verification
1. Server starts and loads checkpoint without errors
2. Test client sends dummy observation, receives valid 14-dim actions
3. End-to-end: real camera frames → server → valid joint commands

---

## Subtask Dataset Preparation

### How Ke-Wang's subtask dataset works

Reference dataset: `KeWangRobotics/libero_10_subtasks` (LeRobot v3.0)

It does NOT use LeRobot's native `subtask_index` + `meta/subtasks.parquet`. Instead, it stores **literal string columns** in the parquet data:
- `subtask` — low-level instruction string (e.g., `"Turn on flat_stove_1"`)
- `overall_task` — high-level task string (e.g., `"turn on the stove and put the moka pot on it"`)
- `task_index` — integer for grouping

OpenPI's `LiberoSubtaskInputs` reads these directly: `data["subtask"]` → `low_prompt`, `data["task"]` → `high_prompt`.

### Your current situation

Your datasets are already separated by subtask — each dataset IS one subtask, but labeled as "task":
- `thewisp/pickup_cylinder_feb_22` → all frames = "pick up the cylinder" (stored as `task`)
- `thewisp/cylinder_socket_merged_head` → all frames = "insert cylinder into socket" (stored as `task`)

### Merge script: tasks → subtasks

**Create**: `scripts/merge_subtask_datasets.py` (in LeRobot or standalone)

Following the same pattern as `KeWangRobotics/libero_10_subtasks`, the script:
1. Takes N source datasets + a high-level task description as input
2. Loads each source dataset's parquet data
3. Adds a `subtask` string column = source dataset's original task text
4. Adds an `overall_task` string column = the high-level description
5. Merges all episodes, re-indexes, writes as a single LeRobot v3.0 dataset
6. Declares `subtask` and `overall_task` as features in `meta/info.json`

```bash
python scripts/merge_subtask_datasets.py \
  --sources thewisp/pickup_cylinder_feb_22 thewisp/cylinder_socket_merged_head \
  --overall-task "assemble cylinder into socket" \
  --output thewisp/cylinder_assembly_subtask
```

Every frame in a source dataset gets the same `subtask` string — no temporal segmentation needed since each source is already one subtask.

Note: LeRobot's existing `aggregate_datasets()` does NOT handle subtask columns, so this is a custom script.

---

## Datasets

- **cylinder_socket**: `thewisp/cylinder_socket_merged_head` — 471 episodes, 64k frames
- **pickup_socket**: `thewisp/pickup_socket_merged_head` — 259 episodes, 60k frames
- Both: `bi_so107_follower`, 14 DoF, 4 cameras (front/left_wrist/right_wrist/top), 720x1280, 30fps, v3.0 format
- Single task per dataset, no subtask annotations yet

---

## Files to Create/Modify

**In forked `openpi_subtask/`:**

| File | Action | Purpose |
|---|---|---|
| `src/openpi/policies/soarm_policy.py` | Create | SO107 key mapping + optional subtask prompt handling |
| `src/openpi/training/config.py` | Modify | Add `LeRobotSOARMDataConfig` + `soarm_pi05_flow` config |
| Robot client script | Create | Websocket client for deployment |

**In LeRobot (or standalone):**

| File | Action | Purpose |
|---|---|---|
| `scripts/merge_subtask_datasets.py` | Create | Merge N source datasets, demote task→subtask, assign high-level task |
