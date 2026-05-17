#!/usr/bin/env python3
"""Evaluate real-time description outputs for DuplexEval.

This script evaluates two metrics only:
1. Temporal sensitivity: whether each sentence describes the right video content at the right time.
2. Content accuracy: whether the full response is factually correct for the entire video.

Expression fluency is intentionally not included.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Tuple

from llm_client import OpenAILLMClient, extract_message_text


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Dict[str, Any], path: str) -> None:
    output_dir = os.path.dirname(os.path.abspath(path))
    os.makedirs(output_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def get_video_duration(video_path: str) -> float:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, check=True)
        return float(result.stdout.strip())
    except Exception as exc:
        print(f"Failed to read video duration for {video_path}: {exc}")
        return 0.0


def normalize_response_items(raw: Any) -> List[Dict[str, Any]]:
    """Normalize common CTC formats to sentence/start/end dictionaries."""

    if isinstance(raw, dict) and isinstance(raw.get("sentences"), list):
        raw_items = raw["sentences"]
    elif isinstance(raw, dict) and isinstance(raw.get("chunks"), list):
        raw_items = raw["chunks"]
    elif isinstance(raw, list):
        raw_items = raw
    else:
        raise ValueError("Response JSON must be a list or contain a 'sentences' or 'chunks' list.")

    normalized: List[Dict[str, Any]] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        text = item.get("sentence", item.get("text", item.get("content", "")))
        if not isinstance(text, str) or not text.strip():
            continue

        if "start" in item:
            start = float(item.get("start") or 0.0)
        else:
            start = float(item.get("current_time") or item.get("time") or 0.0)

        if "end" in item:
            end = float(item.get("end") or start)
        else:
            end = float(item.get("current_time") or item.get("time") or start)

        if end < start:
            end = start

        normalized.append({"sentence": text.strip(), "start": start, "end": end})
    return normalized


def load_response_sentences(path: str) -> List[Dict[str, Any]]:
    sentences = normalize_response_items(read_json(path))
    print(f"Loaded {len(sentences)} response segments from {path}")
    return sentences


def combine_sentences_to_text(sentences: Iterable[Dict[str, Any]]) -> str:
    return " ".join(item.get("sentence", "").strip() for item in sentences if item.get("sentence")).strip()


def load_reference_texts(paths: Optional[List[str]], inline_texts: Optional[List[str]]) -> List[str]:
    references: List[str] = []

    if inline_texts:
        references.extend(text.strip() for text in inline_texts if isinstance(text, str) and text.strip())

    for path in paths or []:
        if not path:
            continue
        if not os.path.exists(path):
            print(f"Reference file not found: {path}")
            continue

        try:
            raw = read_json(path)
        except json.JSONDecodeError:
            with open(path, "r", encoding="utf-8") as handle:
                text = handle.read().strip()
            if text:
                references.append(text)
            continue

        if isinstance(raw, str) and raw.strip():
            references.append(raw.strip())
        elif isinstance(raw, dict):
            for key in ("text", "answer", "content", "annotation", "response"):
                value = raw.get(key)
                if isinstance(value, str) and value.strip():
                    references.append(value.strip())
                    break
        elif isinstance(raw, list):
            if all(isinstance(item, dict) and "sentence" in item for item in raw):
                text = " ".join(str(item.get("sentence", "")).strip() for item in raw).strip()
                if text:
                    references.append(text)
            else:
                references.extend(item.strip() for item in raw if isinstance(item, str) and item.strip())

    deduplicated: List[str] = []
    seen = set()
    for text in references:
        if text not in seen:
            seen.add(text)
            deduplicated.append(text)
    return deduplicated


def encode_image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("utf-8")


def parse_evaluation_response(response: str) -> Dict[str, Any]:
    patterns = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, flags=re.DOTALL)
    for match in patterns:
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", match))
            if "temporal_score" in parsed:
                parsed["temporal_score"] = int(float(parsed["temporal_score"]))
            if "content_score" in parsed:
                parsed["content_score"] = round(float(parsed["content_score"]), 2)
            if "is_relevant" in parsed:
                parsed["is_relevant"] = int(float(parsed["is_relevant"]))
            return parsed
        except Exception:
            continue

    parsed: Dict[str, Any] = {}
    score_patterns = [
        (r'"temporal_score"\s*:\s*(\d+)', "temporal_score", int),
        (r'"is_relevant"\s*:\s*(\d+)', "is_relevant", int),
        (r'"content_score"\s*:\s*(\d+(?:\.\d+)?)', "content_score", float),
    ]
    for pattern, key, caster in score_patterns:
        match = re.search(pattern, response)
        if match:
            value = caster(match.group(1))
            parsed[key] = round(value, 2) if caster is float else value

    for pattern, key in [
        (r'"temporal_reasoning"\s*:\s*"([^"]*)"', "temporal_reasoning"),
        (r'"content_reasoning"\s*:\s*"([^"]*)"', "content_reasoning"),
        (r'"reasoning"\s*:\s*"([^"]*)"', "reasoning"),
    ]:
        match = re.search(pattern, response, flags=re.DOTALL)
        if match:
            parsed[key] = match.group(1).strip()
            break
    return parsed


def extract_frames_for_time_range(
    video_path: str,
    start_time: float,
    end_time: float,
    frames_per_second: int = 2,
) -> List[bytes]:
    if end_time <= start_time:
        return []

    segment_duration = end_time - start_time
    frame_count = max(1, int(segment_duration * frames_per_second))
    frames: List[bytes] = []

    with tempfile.TemporaryDirectory(prefix="Omni-DuplexEval_frames_") as temp_dir:
        for frame_index in range(frame_count):
            if frame_count == 1:
                frame_time = start_time + segment_duration / 2
            else:
                frame_time = start_time + frame_index * segment_duration / (frame_count - 1)
            frame_time = min(frame_time, max(start_time, end_time - 0.01))
            frame_path = os.path.join(temp_dir, f"frame_{frame_index:03d}.jpg")
            command = [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-ss",
                str(frame_time),
                "-i",
                video_path,
                "-vframes",
                "1",
                "-q:v",
                "2",
                "-y",
                frame_path,
            ]
            try:
                subprocess.run(command, capture_output=True, text=True, check=True)
                if os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
                    with open(frame_path, "rb") as handle:
                        frames.append(handle.read())
            except Exception as exc:
                print(f"Frame extraction failed at {frame_time:.2f}s: {exc}")
    return frames


def generate_temporal_window(sentence_start: float, sentence_end: float, video_duration: float) -> Optional[Tuple[float, float]]:
    window_start = max(0.0, sentence_start - 2.0)
    window_end = max(window_start + 0.5, sentence_end - 2.0)
    if window_end <= video_duration and window_end - window_start >= 0.5:
        return window_start, window_end
    return None


def build_temporal_prompt(start_time: float, end_time: float, response_text: str, question: str) -> str:
    return f"""
You are evaluating a real-time video description system.

Basic information:
- Video segment: {start_time:.2f}s to {end_time:.2f}s.
- User instruction: {question}
- Model response sentence: "{response_text}"

Evaluation principles:
- Judge only the current video segment.
- Ignore response content that clearly belongs to earlier or later segments unless it contradicts the current segment.
- Language does not matter; evaluate semantic correctness and temporal alignment only.
- Decide whether the sentence is a substantive response or only a filler/polite phrase.

Temporal sensitivity score:
- 3: Excellent temporal alignment. The sentence accurately describes the current segment and follows the instruction.
- 2: Mostly aligned. Minor inaccuracies, omissions, or harmless references to nearby segments are allowed.
- 1: Poor alignment. Major inaccuracies, wrong timing, or mostly irrelevant content.
- 0: No meaningful alignment with the current segment.

Relevance label:
- 1: The sentence contains substantive task- or video-related information.
- 0: The sentence is only filler, acknowledgement, meta-commentary, or generic text without useful content.

Return exactly this JSON object:
{{
  "temporal_score": <0, 1, 2, or 3>,
  "temporal_reasoning": "<brief explanation>",
  "is_relevant": <0 or 1>
}}
"""


def evaluate_temporal_sentence(
    sentence: Dict[str, Any],
    video_path: str,
    video_duration: float,
    question: str,
    model_id: str,
    frames_per_second: int,
) -> Dict[str, Any]:
    sentence_text = sentence["sentence"]
    sentence_start = float(sentence["start"])
    sentence_end = float(sentence["end"])
    window = generate_temporal_window(sentence_start, sentence_end, video_duration)
    if not window:
        return {
            "sentence": sentence_text,
            "sentence_start": sentence_start,
            "sentence_end": sentence_end,
            "error": "No valid temporal window",
            "temporal_score": 0,
            "is_relevant": 0,
            "window_start": None,
            "window_end": None,
            "frame_count": 0,
        }

    window_start, window_end = window
    frame_bytes = extract_frames_for_time_range(video_path, window_start, window_end, frames_per_second)
    if not frame_bytes:
        return {
            "sentence": sentence_text,
            "sentence_start": sentence_start,
            "sentence_end": sentence_end,
            "error": "No frames extracted",
            "temporal_score": 0,
            "is_relevant": 0,
            "window_start": window_start,
            "window_end": window_end,
            "frame_count": 0,
        }

    prompt = build_temporal_prompt(window_start, window_end, sentence_text, question)
    content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
    for frame in frame_bytes:
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{encode_image_to_base64(frame)}"},
            }
        )

    client = OpenAILLMClient()
    response = client.chat_completion(
        messages=[
            {
                "role": "system",
                "content": "You are a careful evaluator for real-time multimodal systems.",
            },
            {"role": "user", "content": content},
        ],
        model=model_id,
        temperature=0.1,
        max_tokens=800,
    )

    if "error" in response:
        return {
            "sentence": sentence_text,
            "sentence_start": sentence_start,
            "sentence_end": sentence_end,
            "error": response.get("error"),
            "temporal_score": 0,
            "is_relevant": 0,
            "window_start": window_start,
            "window_end": window_end,
            "frame_count": len(frame_bytes),
        }

    parsed = parse_evaluation_response(extract_message_text(response))
    return {
        "sentence": sentence_text,
        "sentence_start": sentence_start,
        "sentence_end": sentence_end,
        "sentence_duration": max(0.0, sentence_end - sentence_start),
        "window_start": window_start,
        "window_end": window_end,
        "temporal_score": int(parsed.get("temporal_score", 0)),
        "is_relevant": int(parsed.get("is_relevant", 0)),
        "temporal_reasoning": parsed.get("temporal_reasoning", parsed.get("reasoning", "")),
        "frame_count": len(frame_bytes),
        "error": None,
    }


def summarize_temporal_results(sentence_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    valid = [item for item in sentence_results if item.get("error") in (None, "")]
    relevant = [item for item in valid if item.get("is_relevant", 0) == 1]
    irrelevant = [item for item in valid if item.get("is_relevant", 0) == 0]
    relevant_scores = [int(item.get("temporal_score", 0)) for item in relevant]
    avg_score = sum(relevant_scores) / len(relevant_scores) if relevant_scores else 0.0

    total_valid_duration = sum(float(item.get("sentence_duration", 0.0)) for item in valid)
    irrelevant_duration = sum(float(item.get("sentence_duration", 0.0)) for item in irrelevant)

    return {
        "avg_temporal_score": round(avg_score, 4),
        "total_sentences": len(sentence_results),
        "evaluated_sentences": len(valid),
        "error_count": len(sentence_results) - len(valid),
        "relevant_sentences_count": len(relevant),
        "irrelevant_sentences_count": len(irrelevant),
        "score_distribution": {
            "3_points": sum(1 for score in relevant_scores if score == 3),
            "2_points": sum(1 for score in relevant_scores if score == 2),
            "1_point": sum(1 for score in relevant_scores if score == 1),
            "0_points": sum(1 for score in relevant_scores if score == 0),
        },
        "relevance_stats": {
            "first_relevant_time": min((item.get("sentence_start") for item in relevant), default=None),
            "total_relevant_duration": round(
                sum(float(item.get("sentence_duration", 0.0)) for item in relevant), 4
            ),
            "total_irrelevant_duration": round(irrelevant_duration, 4),
            "irrelevant_duration_ratio": round(
                irrelevant_duration / total_valid_duration if total_valid_duration > 0 else 0.0, 4
            ),
        },
        "excluded_irrelevant_sentences": [
            {
                "sentence": item.get("sentence", ""),
                "start": item.get("sentence_start"),
                "end": item.get("sentence_end"),
                "temporal_score": item.get("temporal_score", 0),
            }
            for item in irrelevant
        ],
    }


def run_temporal_evaluation(
    sentences: List[Dict[str, Any]],
    video_path: str,
    video_duration: float,
    question: str,
    model_id: str,
    frames_per_second: int,
    max_workers: int,
) -> Dict[str, Any]:
    results: List[Dict[str, Any]] = []
    print(f"Running temporal sensitivity on {len(sentences)} segments with {max_workers} workers.")

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(
                evaluate_temporal_sentence,
                sentence,
                video_path,
                video_duration,
                question,
                model_id,
                frames_per_second,
            )
            for sentence in sentences
        ]
        for index, future in enumerate(as_completed(futures), 1):
            try:
                results.append(future.result(timeout=900))
            except Exception as exc:
                results.append({"error": str(exc), "temporal_score": 0, "is_relevant": 0})
            print(f"Temporal progress: {index}/{len(futures)}")

    results.sort(key=lambda item: (item.get("sentence_start") is None, item.get("sentence_start") or 0.0))
    return {"sentence_results": results, "summary": summarize_temporal_results(results)}


def build_content_prompt(response_text: str, question: str, references: List[str]) -> str:
    reference_block = ""
    if references:
        reference_block = "Reference annotations, for reference only:\n"
        for index, reference in enumerate(references, 1):
            reference_block += f"{index}. {reference}\n"

    return f"""
You are a precise content-accuracy evaluator for video description.

Basic information:
- User instruction: {question}
- Model response: "{response_text}"
{reference_block}

Evaluation goal:
Judge whether the model response is factually accurate for the whole video and aligned with the instruction.
Consider object/action/color/count/spatial/event errors, hallucinations, omissions, and irrelevant content.
Reference annotations are optional guidance; the primary evidence is the video itself.

Scoring:
- Use a decimal score from 0.00 to 3.00.
- Start from 3.00 and deduct for each error.
- Use 0.00 only when the response is empty, completely irrelevant, or contains no correct video facts.
- Output exactly two decimal places.

Return exactly this JSON object:
{{
  "content_score": <decimal from 0.00 to 3.00>,
  "content_reasoning": "<brief explanation with main errors and final score>"
}}
"""


def evaluate_content_accuracy(
    client: OpenAILLMClient,
    video_path: str,
    response_text: str,
    question: str,
    references: List[str],
    model_id: str,
) -> Dict[str, Any]:
    prompt = build_content_prompt(response_text, question, references)
    response = client.chat_completion(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "video", "video": video_path},
                ],
            }
        ],
        model=model_id,
        temperature=0.1,
        max_tokens=1200,
    )

    if "error" in response:
        return {"score": 0.0, "success": False, "error": response.get("error")}

    parsed = parse_evaluation_response(extract_message_text(response))
    return {
        "score": round(float(parsed.get("content_score", 0.0)), 2),
        "reasoning": parsed.get("content_reasoning", parsed.get("reasoning", "")),
        "success": True,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DuplexEval real-time description outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--video", required=True, help="Path to the video file.")
    parser.add_argument("--input", required=True, help="Path to the model response JSON file.")
    parser.add_argument("--question", required=True, help="Original user instruction.")
    parser.add_argument("--output", default="real_time_description_result.json", help="Output JSON path.")
    parser.add_argument("--gt", nargs="*", default=None, help="Optional reference annotation file paths.")
    parser.add_argument("--gt-text", nargs="*", default=None, help="Optional reference annotation text strings.")
    parser.add_argument(
        "--model",
        default=os.environ.get("DUPLEXEVAL_MODEL", "EVALUATOR_MODEL"),
        help="Evaluator model id. Defaults to DUPLEXEVAL_MODEL when set.",
    )
    parser.add_argument("--metrics", nargs="+", choices=["temporal", "content", "all"], default=["all"])
    parser.add_argument("--fps", type=int, default=2, help="Frames per second for temporal windows.")
    parser.add_argument("--max-workers", type=int, default=8, help="Parallel workers for temporal evaluation.")
    parser.add_argument("--no-save", action="store_true", help="Print results without writing the output JSON.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    start_time = time.time()
    metrics = ["temporal", "content"] if "all" in args.metrics else args.metrics

    print("=" * 72)
    print("DuplexEval Real-Time Description Evaluation")
    print("=" * 72)
    print(f"Video: {args.video}")
    print(f"Response: {args.input}")
    print(f"Output: {args.output}")
    print(f"Metrics: {metrics}")

    try:
        sentences = load_response_sentences(args.input)
        response_text = combine_sentences_to_text(sentences)
        references = load_reference_texts(args.gt, args.gt_text)
        video_duration = get_video_duration(args.video)

        results: Dict[str, Any] = {
            "metadata": {
                "script": "real_time_description.py",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "video_path": args.video,
                "response_path": args.input,
                "question": args.question,
                "model_id": args.model,
                "metrics_evaluated": metrics,
                "frames_per_second": args.fps,
                "max_workers": args.max_workers,
                "video_duration": video_duration,
            },
            "input_summary": {
                "total_segments": len(sentences),
                "full_response": response_text,
                "reference_count": len(references),
                "references": references,
            },
        }

        client = OpenAILLMClient()

        if "temporal" in metrics:
            results["temporal_sensitivity"] = run_temporal_evaluation(
                sentences=sentences,
                video_path=args.video,
                video_duration=video_duration,
                question=args.question,
                model_id=args.model,
                frames_per_second=args.fps,
                max_workers=args.max_workers,
            )

        if "content" in metrics:
            print("Running content accuracy evaluation.")
            results["content_accuracy"] = evaluate_content_accuracy(
                client=client,
                video_path=args.video,
                response_text=response_text,
                question=args.question,
                references=references,
                model_id=args.model,
            )

        results["evaluation_time_seconds"] = round(time.time() - start_time, 2)

        if not args.no_save:
            write_json(results, args.output)
            print(f"Saved results to {args.output}")

        if "temporal_sensitivity" in results:
            score = results["temporal_sensitivity"]["summary"]["avg_temporal_score"]
            print(f"Temporal sensitivity: {score:.2f}/3.00")
        if "content_accuracy" in results and results["content_accuracy"].get("success"):
            print(f"Content accuracy: {results['content_accuracy']['score']:.2f}/3.00")
        print(f"Total time: {results['evaluation_time_seconds']:.2f}s")
        return 0

    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
