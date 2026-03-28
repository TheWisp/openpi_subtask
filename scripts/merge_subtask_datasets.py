#!/usr/bin/env python3
"""Merge multiple LeRobot datasets into a single subtask-annotated dataset.

Each source dataset represents one subtask. The script:
1. Merges all source datasets into one using LeRobot's merge_datasets()
2. Adds a "subtask" string column = source dataset's original task text
3. Adds a "task" string column = user-provided high-level task description

The output follows the same format as KeWangRobotics/libero_10_subtasks,
with literal "task" and "subtask" string columns in the parquet data.

Usage:
    python scripts/merge_subtask_datasets.py \
        --sources thewisp/pickup_cylinder_feb_22 thewisp/cylinder_socket_merged_head \
        --subtask-names "pick up the cylinder" "insert cylinder into socket" \
        --overall-task "assemble cylinder into socket" \
        --output thewisp/cylinder_assembly_subtask

    If --subtask-names is omitted, uses each source dataset's original task text.
"""

import argparse
import shutil
from pathlib import Path

from lerobot.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
from lerobot.datasets.dataset_tools import merge_datasets, modify_features


def main():
    parser = argparse.ArgumentParser(
        description="Merge LeRobot datasets with subtask annotations"
    )
    parser.add_argument(
        "--sources", nargs="+", required=True,
        help="Source dataset repo_ids (each is one subtask)",
    )
    parser.add_argument(
        "--overall-task", required=True,
        help="High-level task description",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output dataset repo_id",
    )
    parser.add_argument(
        "--subtask-names", nargs="+", default=None,
        help="Custom subtask names, one per source (default: use each dataset's task text)",
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: HF_LEROBOT_HOME/output)",
    )
    args = parser.parse_args()

    if args.subtask_names and len(args.subtask_names) != len(args.sources):
        parser.error(f"--subtask-names must match --sources count ({len(args.sources)})")

    output_dir = (
        Path(args.output_dir) if args.output_dir
        else HF_LEROBOT_HOME / args.output
    )

    # Load and verify source datasets
    print(f"Loading {len(args.sources)} source datasets...")
    datasets = []
    for repo_id in args.sources:
        print(f"  Loading {repo_id}...")
        ds = LeRobotDataset(repo_id)
        # Verify integrity before merging — corrupted sources propagate silently
        from lerobot.datasets.dataset_tools import verify_dataset
        result = verify_dataset(ds.root, check_videos=False, verbose=False)
        if not result.is_valid:
            raise RuntimeError(
                f"Source dataset {repo_id} failed verification with {len(result.errors)} error(s):\n"
                + "\n".join(f"  - {e}" for e in result.errors)
            )
        datasets.append(ds)

    # Build original_task_name -> custom_subtask_name mapping from sources.
    # We map by name (not index) because aggregate_datasets remaps indices.
    source_task_names = []
    for ds in datasets:
        # ds.meta.tasks is a DataFrame: index=task_name, columns=["task_index"]
        names = list(ds.meta.tasks.index)
        source_task_names.append(names)

    # Map: original_task_name -> subtask_name (custom or original)
    name_to_subtask = {}
    for i, names in enumerate(source_task_names):
        for name in names:
            if args.subtask_names:
                name_to_subtask[name] = args.subtask_names[i]
            else:
                name_to_subtask[name] = name

    print(f"Source task name -> subtask mapping: {name_to_subtask}")

    # Merge datasets
    merge_dir = output_dir.parent / f"{output_dir.name}_merge_tmp"
    print(f"Merging into {merge_dir}...")
    merged = merge_datasets(
        datasets,
        output_repo_id=f"{args.output}_tmp",
        output_dir=merge_dir,
    )
    print(
        f"Merged: {merged.meta.total_episodes} episodes, "
        f"{merged.meta.total_frames} frames"
    )

    # Sanity check: merged episode/frame counts must equal sum of sources
    expected_episodes = sum(ds.meta.total_episodes for ds in datasets)
    expected_frames = sum(ds.meta.total_frames for ds in datasets)
    if merged.meta.total_episodes != expected_episodes:
        raise RuntimeError(
            f"Merged episode count ({merged.meta.total_episodes}) != "
            f"sum of sources ({expected_episodes}). "
            f"Source datasets may have corrupted metadata — run verify_dataset on each."
        )
    if merged.meta.total_frames != expected_frames:
        raise RuntimeError(
            f"Merged frame count ({merged.meta.total_frames}) != "
            f"sum of sources ({expected_frames}). "
            f"Source datasets may have corrupted metadata — run verify_dataset on each."
        )

    # Build task_index -> subtask_name from the MERGED dataset's tasks.
    # This uses the actual remapped task indices, not an offset guess.
    task_map = {}
    for task_name, row in merged.meta.tasks.iterrows():
        tidx = int(row["task_index"])
        task_map[tidx] = name_to_subtask.get(task_name, task_name)
    print(f"Merged task_index -> subtask: {task_map}")

    # Add subtask and task string columns
    print("Adding subtask and task columns...")
    overall_task = args.overall_task

    def subtask_fn(row, ep_idx, frame_idx):
        tidx = int(row["task_index"])
        return task_map.get(tidx, f"unknown_task_{tidx}")

    def task_fn(row, ep_idx, frame_idx):
        return overall_task

    result = modify_features(
        merged,
        add_features={
            "subtask": (
                subtask_fn,
                {"dtype": "string", "shape": (1,), "names": None},
            ),
            "task": (
                task_fn,
                {"dtype": "string", "shape": (1,), "names": None},
            ),
        },
        output_dir=output_dir,
        repo_id=args.output,
    )

    # Fix task_index consistency: the merged data has task_index 0,1,2,...
    # but we want a single overall task. Set all task_index to 0 in data parquet,
    # overwrite meta/tasks.parquet, and fix info.json.
    import glob
    import json
    import pandas as pd

    # 1. Set all task_index to 0 in data parquet files
    data_files = sorted(glob.glob(str(output_dir / "data" / "**" / "*.parquet"), recursive=True))
    for f in data_files:
        df = pd.read_parquet(f)
        if (df["task_index"] != 0).any():
            df["task_index"] = 0
            df.to_parquet(f)
    print(f"Set all task_index to 0 in {len(data_files)} data file(s)")

    # 2. Fix "tasks" column in meta/episodes parquet files (old task names -> overall task)
    ep_files = sorted(glob.glob(str(output_dir / "meta" / "episodes" / "**" / "*.parquet"), recursive=True))
    for f in ep_files:
        df = pd.read_parquet(f)
        if "tasks" in df.columns:
            df["tasks"] = [[overall_task]] * len(df)
            df.to_parquet(f)
    print(f"Updated 'tasks' column in {len(ep_files)} episodes file(s)")

    # 3. Overwrite meta/tasks.parquet with single overall task
    new_tasks = pd.DataFrame({"task_index": [0]}, index=[overall_task])
    new_tasks.to_parquet(output_dir / "meta" / "tasks.parquet")
    print(f"Updated meta/tasks.parquet -> {overall_task!r}")

    # 4. Fix info.json total_tasks
    info_path = output_dir / "meta" / "info.json"
    with open(info_path) as f:
        info = json.load(f)
    info["total_tasks"] = 1
    with open(info_path, "w") as f:
        json.dump(info, f, indent=4)
    print(f"Updated info.json total_tasks -> 1")

    # Verify output dataset integrity
    print("\nVerifying output dataset...")
    output_verification = verify_dataset(output_dir, check_videos=False, verbose=False)
    if not output_verification.is_valid:
        print(f"WARNING: Output dataset has {len(output_verification.errors)} error(s):")
        for e in output_verification.errors:
            print(f"  - {e}")
        raise RuntimeError("Output dataset failed verification — aborting")
    print("Output dataset verified OK")

    print(f"\nOutput: {args.output} at {output_dir}")
    print(f"  Episodes: {result.meta.total_episodes}")
    print(f"  Frames: {result.meta.total_frames}")
    print(f"  Features: {list(result.meta.features.keys())}")
    print(f"  Subtasks: {dict(sorted(task_map.items()))}")

    if merge_dir.exists():
        shutil.rmtree(merge_dir)
        print(f"Cleaned up {merge_dir}")

    print("\nUse with OpenPI:")
    print(f"  uv run scripts/train.py soarm_pi05_flow \\")
    print(f"    --data.repo-id={args.output} \\")
    print(f"    --data.prompt-from-task=False")


if __name__ == "__main__":
    main()
