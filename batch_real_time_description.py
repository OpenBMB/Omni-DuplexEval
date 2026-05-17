#!/usr/bin/env python3
"""Batch evaluation for DuplexEval real-time description splits."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

from data_utils import RTD_SPLITS, cleanup_media_dir, materialize_video, non_empty_strings, resolve_response_path


SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "real_time_description.py")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch real-time description evaluation on HuggingFace dataset splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="foragi/Omni-DuplexEval-Examples",
        help="HuggingFace dataset name or local dataset path.",
    )
    parser.add_argument("--splits", nargs="+", default=RTD_SPLITS, help="Dataset splits to evaluate.")
    parser.add_argument("--response-root", required=True, help="Root directory containing model response JSON files.")
    parser.add_argument(
        "--response-template",
        default=None,
        help="Optional path template, e.g. '{response_root}/{split}/{id}.json'.",
    )
    parser.add_argument("--output-root", required=True, help="Root directory for evaluation outputs.")
    parser.add_argument(
        "--model",
        default=os.environ.get("DUPLEXEVAL_MODEL", "EVALUATOR_MODEL"),
        help="Evaluator model id. Defaults to DUPLEXEVAL_MODEL when set.",
    )
    parser.add_argument("--metrics", nargs="+", choices=["temporal", "content", "all"], default=["all"])
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for temporal evaluation.")
    parser.add_argument("--eval-workers", type=int, default=8, help="Workers inside each single-sample evaluation.")
    parser.add_argument("--sample-workers", type=int, default=2, help="Number of samples evaluated in parallel.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows per split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running evaluation.")
    parser.add_argument("--keep-media", action="store_true", help="Keep extracted media files under output-root.")
    return parser.parse_args()


def load_split(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the 'datasets' package to use the batch scripts.") from exc

    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def output_path(output_root: str, split: str, sample_id: str) -> str:
    path = Path(output_root) / split / f"{sample_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def build_command(args: argparse.Namespace, row: Dict[str, Any], split: str, media_dir: str) -> List[str]:
    sample_id = str(row["id"])
    response_path = resolve_response_path(args.response_root, split, sample_id, args.response_template)
    video_path = materialize_video(row["video"], media_dir, f"{split}_{sample_id}")
    references = non_empty_strings([row.get("answer1"), row.get("answer2")])

    command = [
        sys.executable,
        SCRIPT_PATH,
        "--video",
        video_path,
        "--input",
        response_path,
        "--question",
        str(row.get("question_text", "")),
        "--output",
        output_path(args.output_root, split, sample_id),
        "--model",
        args.model,
        "--fps",
        str(args.fps),
        "--max-workers",
        str(args.eval_workers),
        "--metrics",
        *args.metrics,
    ]
    if references:
        command.extend(["--gt-text", *references])
    return command


def process_row(args: argparse.Namespace, row: Dict[str, Any], split: str, media_dir: str) -> Dict[str, Any]:
    sample_id = str(row["id"])
    out_path = output_path(args.output_root, split, sample_id)
    response_path = resolve_response_path(args.response_root, split, sample_id, args.response_template)

    if not os.path.exists(response_path):
        return {
            "split": split,
            "id": sample_id,
            "status": "skipped",
            "reason": f"Response file not found: {response_path}",
        }
    if os.path.exists(out_path) and not args.overwrite:
        return {"split": split, "id": sample_id, "status": "skipped", "reason": "Output already exists"}

    command = build_command(args, row, split, media_dir)
    if args.dry_run:
        return {"split": split, "id": sample_id, "status": "dry_run", "command": " ".join(command)}

    start = time.time()
    result = subprocess.run(command, capture_output=True, text=True)
    elapsed = round(time.time() - start, 2)
    if result.returncode == 0:
        return {
            "split": split,
            "id": sample_id,
            "status": "success",
            "output": out_path,
            "elapsed_seconds": elapsed,
            "stdout_tail": result.stdout[-1000:],
        }
    return {
        "split": split,
        "id": sample_id,
        "status": "failed",
        "returncode": result.returncode,
        "elapsed_seconds": elapsed,
        "stderr_tail": result.stderr[-2000:],
        "stdout_tail": result.stdout[-1000:],
    }


def main() -> int:
    args = parse_arguments()
    Path(args.output_root).mkdir(parents=True, exist_ok=True)
    media_dir = os.path.join(args.output_root, "_media")
    start_time = time.time()

    print("=" * 72)
    print("DuplexEval Batch Real-Time Description Evaluation")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {args.splits}")
    print(f"Response root: {args.response_root}")
    print(f"Output root: {args.output_root}")

    results: List[Dict[str, Any]] = []
    try:
        tasks = []
        for split in args.splits:
            rows = load_split(args.dataset, split)
            if args.limit is not None:
                rows = rows[: args.limit]
            for row in rows:
                tasks.append((row, split))

        print(f"Total rows: {len(tasks)}")
        with ThreadPoolExecutor(max_workers=args.sample_workers) as executor:
            future_to_task = {
                executor.submit(process_row, args, row, split, media_dir): (split, str(row["id"]))
                for row, split in tasks
            }
            for index, future in enumerate(as_completed(future_to_task), 1):
                split, sample_id = future_to_task[future]
                try:
                    item = future.result()
                except Exception as exc:
                    item = {"split": split, "id": sample_id, "status": "failed", "error": str(exc)}
                results.append(item)
                print(f"[{index}/{len(tasks)}] {item['split']}/{item['id']}: {item['status']}")

        status_counts: Dict[str, int] = {}
        for item in results:
            status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1

        summary = {
            "metadata": {
                "script": "batch_real_time_description.py",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "elapsed_seconds": round(time.time() - start_time, 2),
                "args": vars(args),
            },
            "status_counts": status_counts,
            "results": results,
        }
        summary_path = os.path.join(args.output_root, "batch_real_time_description_summary.json")
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, ensure_ascii=False, indent=2)
        print(f"Saved summary to {summary_path}")
        return 0 if status_counts.get("failed", 0) == 0 else 1
    finally:
        cleanup_media_dir(media_dir, args.keep_media)


if __name__ == "__main__":
    sys.exit(main())
