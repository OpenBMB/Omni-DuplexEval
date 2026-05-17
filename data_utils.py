#!/usr/bin/env python3
"""Shared utilities for DuplexEval batch scripts."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Iterable, List, Optional


RTD_SPLITS = [
    "RTD_world_knowledge",
    "RTD_counting",
    "RTD_fine_grained_movement",
    "RTD_interaction_relation",
    "RTD_OCR",
    "RTD_Omni",
]

PR_SPLITS = [
    "PR_correction",
    "PR_event_reminder",
    "PR_post_event_reminder",
]


def non_empty_strings(values: Iterable[Any]) -> List[str]:
    output = []
    for value in values:
        if value is None:
            continue
        text = str(value).strip()
        if text:
            output.append(text)
    return output


def maybe_float_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return str(float(value))
    if isinstance(value, str) and value.strip():
        try:
            return str(float(value.strip()))
        except ValueError:
            return None
    return None


def resolve_response_path(response_root: str, split: str, sample_id: str, template: Optional[str] = None) -> str:
    values = {"response_root": response_root, "split": split, "id": sample_id}
    candidates = []
    if template:
        candidates.append(template.format(**values))
    candidates.extend(
        [
            os.path.join(response_root, split, f"{sample_id}.json"),
            os.path.join(response_root, f"{sample_id}.json"),
            os.path.join(response_root, split.lower(), f"{sample_id}.json"),
        ]
    )
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return candidates[0]


def materialize_video(video_value: Any, output_dir: str, sample_id: str) -> str:
    """Return a local video path from common HuggingFace video representations."""

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if isinstance(video_value, str):
        return video_value

    if isinstance(video_value, dict):
        path = video_value.get("path")
        if path and os.path.exists(path):
            return str(path)

        bytes_value = video_value.get("bytes")
        if bytes_value:
            output_path = os.path.join(output_dir, f"{sample_id}.mp4")
            with open(output_path, "wb") as handle:
                handle.write(bytes_value)
            return output_path

    path_attr = getattr(video_value, "path", None)
    if path_attr and os.path.exists(path_attr):
        return str(path_attr)

    if isinstance(video_value, (bytes, bytearray)):
        output_path = os.path.join(output_dir, f"{sample_id}.mp4")
        with open(output_path, "wb") as handle:
            handle.write(video_value)
        return output_path

    raise ValueError("Could not materialize the HuggingFace video field into a local file.")


def cleanup_media_dir(path: str, keep: bool) -> None:
    if keep:
        return
    if os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)
