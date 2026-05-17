#!/usr/bin/env python3
"""Batch evaluation for DuplexEval proactive reminder splits."""

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

from data_utils import PR_SPLITS, maybe_float_string, resolve_response_path


SCRIPT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "proactive_reminder.py")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run batch proactive reminder evaluation on HuggingFace dataset splits.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset",
        default="foragi/Omni-DuplexEval-Examples",
        help="HuggingFace dataset name or local dataset path.",
    )
    parser.add_argument("--splits", nargs="+", default=PR_SPLITS, help="Dataset splits to evaluate.")
    parser.add_argument("--response-root", required=True, help="Root directory containing model response JSON files.")
    parser.add_argument(
        "--response-template",
        default=None,
        help="Optional path template, e.g. '{response_root}/{split}/{id}.json'.",
    )
    parser.add_argument("--output-root", required=True, help="Root directory for evaluation outputs.")
    parser.add_argument(
        "--model-id",
        default=os.environ.get("DUPLEXEVAL_MODEL", "EVALUATOR_MODEL"),
        help="Evaluator model id. Defaults to DUPLEXEVAL_MODEL when set.",
    )
    parser.add_argument("--window-size", type=float, default=10.0, help="Evaluation window after each reminder time.")
    parser.add_argument("--sample-workers", type=int, default=4, help="Number of samples evaluated in parallel.")
    parser.add_argument("--limit", type=int, default=None, help="Optional maximum number of rows per split.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing output files.")
    parser.add_argument("--dry-run", action="store_true", help="Print commands without running evaluation.")
    return parser.parse_args()


def load_split(dataset_name: str, split: str) -> List[Dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("Install the 'datasets' package to use the batch scripts.") from exc

    dataset = load_dataset(dataset_name, split=split)
    return [dict(row) for row in dataset]


def task_type_from_split(split: str) -> str:
    split_lower = split.lower()
    if "correction" in split_lower:
        return "correction"
    if "post_event" in split_lower:
        return "post_event_reminder"
    return "proactive_reminder"


def output_path(output_root: str, split: str, sample_id: str) -> str:
    path = Path(output_root) / split / f"{sample_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def reminder_times_from_row(row: Dict[str, Any]) -> List[str]:
    times = []
    for key in ("reminder1", "reminder2"):
        value = maybe_float_string(row.get(key))
        if value is not None:
            times.append(value)
    return times


def build_command(args: argparse.Namespace, row: Dict[str, Any], split: str) -> List[str]:
    sample_id = str(row["id"])
    response_path = resolve_response_path(args.response_root, split, sample_id, args.response_template)
    command = [
        sys.executable,
        SCRIPT_PATH,
        "--instruction",
        str(row.get("question_text", "")),
        "--response",
        response_path,
        "--task-type",
        task_type_from_split(split),
        "--output",
        output_path(args.output_root, split, sample_id),
        "--model-id",
        args.model_id,
        "--window-size",
        str(args.window_size),
    ]

    answer = str(row.get("answer1", "") or "").strip()
    if answer:
        command.extend(["--ground-answer", answer])

    reminder_times = reminder_times_from_row(row)
    if reminder_times:
        command.extend(["--reminder-times", *reminder_times])

    return command


def process_row(args: argparse.Namespace, row: Dict[str, Any], split: str) -> Dict[str, Any]:
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

    command = build_command(args, row, split)
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
    start_time = time.time()

    print("=" * 72)
    print("DuplexEval Batch Proactive Reminder Evaluation")
    print("=" * 72)
    print(f"Dataset: {args.dataset}")
    print(f"Splits: {args.splits}")
    print(f"Response root: {args.response_root}")
    print(f"Output root: {args.output_root}")

    tasks = []
    for split in args.splits:
        rows = load_split(args.dataset, split)
        if args.limit is not None:
            rows = rows[: args.limit]
        for row in rows:
            tasks.append((row, split))

    print(f"Total rows: {len(tasks)}")
    results: List[Dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.sample_workers) as executor:
        future_to_task = {
            executor.submit(process_row, args, row, split): (split, str(row["id"])) for row, split in tasks
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
            "script": "batch_proactive_reminder.py",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "elapsed_seconds": round(time.time() - start_time, 2),
            "args": vars(args),
        },
        "status_counts": status_counts,
        "results": results,
    }
    summary_path = os.path.join(args.output_root, "batch_proactive_reminder_summary.json")
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    print(f"Saved summary to {summary_path}")
    return 0 if status_counts.get("failed", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
