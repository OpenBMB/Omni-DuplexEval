#!/usr/bin/env python3
"""Evaluate proactive reminder and correction outputs for DuplexEval."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import traceback
from typing import Any, Dict, List, Optional

from llm_client import OpenAILLMClient, extract_message_text


TASK_ALIASES = {
    "correction": "correction",
    "pr_correction": "correction",
    "event_reminder": "proactive_reminder",
    "proactive_reminder": "proactive_reminder",
    "pr_event_reminder": "proactive_reminder",
    "post_event_reminder": "post_event_reminder",
    "pr_post_event_reminder": "post_event_reminder",
}


def canonical_task_type(task_type: str) -> str:
    key = task_type.strip().lower().replace("-", "_").replace(" ", "_")
    if key not in TASK_ALIASES:
        raise ValueError(
            f"Unsupported task type '{task_type}'. Use correction, proactive_reminder, or post_event_reminder."
        )
    return TASK_ALIASES[key]


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(data: Dict[str, Any], path: str) -> None:
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def normalize_chunks(raw: Any) -> List[Dict[str, Any]]:
    if isinstance(raw, dict) and isinstance(raw.get("chunks"), list):
        items = raw["chunks"]
    elif isinstance(raw, dict) and isinstance(raw.get("sentences"), list):
        items = raw["sentences"]
    elif isinstance(raw, list):
        items = raw
    else:
        raise ValueError("Response JSON must be a list or contain a 'chunks' or 'sentences' list.")

    chunks: List[Dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        text = item.get("text", item.get("sentence", item.get("content", "")))
        if not isinstance(text, str) or not text.strip():
            continue
        current_time = item.get("current_time", item.get("start", item.get("time", 0.0)))
        chunks.append({"text": text.strip(), "current_time": float(current_time or 0.0)})
    return chunks


def response_meta(raw: Any) -> Dict[str, Any]:
    if not isinstance(raw, dict):
        return {"sample_id": "", "dataset": ""}
    return {
        "sample_id": raw.get("id", raw.get("sample_id", "")),
        "dataset": raw.get("dataset", ""),
        "source_task_type": raw.get("task_type", ""),
    }


def parse_float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def load_ground_truth(args: argparse.Namespace) -> Dict[str, Any]:
    gt_data: Dict[str, Any] = {}
    if args.gt:
        raw_gt = read_json(args.gt)
        if not isinstance(raw_gt, dict):
            raise ValueError("Ground-truth JSON must be an object.")
        gt_data.update(raw_gt)

    if args.ground_answer:
        gt_data["answer"] = args.ground_answer

    reminder_times = [parse_float_or_none(value) for value in args.reminder_times or []]
    reminder_times = [value for value in reminder_times if value is not None]
    for index, start_time in enumerate(reminder_times, 1):
        gt_data[f"reminder_{index}"] = {"start": start_time, "end": start_time}

    return gt_data


def collect_reminders(gt_data: Dict[str, Any]) -> List[Dict[str, float]]:
    reminders: List[Dict[str, float]] = []
    for index in range(1, 4):
        for key in (f"reminder_{index}", f"reminder{index}"):
            value = gt_data.get(key)
            start_time: Optional[float]
            end_time: Optional[float]
            if isinstance(value, dict):
                start_time = parse_float_or_none(value.get("start"))
                end_time = parse_float_or_none(value.get("end"))
            else:
                start_time = parse_float_or_none(value)
                end_time = start_time
            if start_time is not None:
                reminders.append(
                    {
                        "index": float(index),
                        "start": start_time,
                        "end": end_time if end_time is not None else start_time,
                    }
                )
                break
    return reminders


def response_text_near_time(chunks: List[Dict[str, Any]], start_time: float, window_size: float) -> str:
    end_time = start_time + window_size
    selected = [
        chunk["text"]
        for chunk in chunks
        if start_time <= float(chunk.get("current_time", 0.0)) <= end_time and chunk.get("text")
    ]
    return " ".join(selected).strip()


def full_response_text(chunks: List[Dict[str, Any]]) -> str:
    return " ".join(chunk.get("text", "") for chunk in chunks if chunk.get("text")).strip()


def build_reminder_prompt(instruction: str, response: str, task_type: str) -> str:
    if task_type == "post_event_reminder":
        timing_note = "The response is evaluated immediately after the target event has occurred."
    else:
        timing_note = "The response is evaluated at the target event time or immediately after it."

    return f"""
You are judging whether a model successfully completed an event-reminder task.

Task description:
The user gave an instruction asking the system to remind them when a specific event happens.
{timing_note}
The provided model text is the response segment generated in the evaluation window.

Input:
- User instruction: {instruction}
- Model response segment: "{response}"

Success criteria:
1. The response clearly refers to the target event in the instruction.
2. The response communicates a reminder, notification, or confirmation that the event has happened.
3. Vague narration, unrelated descriptions, or wrong-event reminders are failures.
4. The response does not need to match the instruction wording exactly if the meaning is clear.

Scoring:
- 1 = successful reminder.
- 0 = unsuccessful reminder.

Return exactly this JSON object:
{{
  "success_score": <0 or 1>,
  "reasoning": "<brief explanation>"
}}
"""


def build_correction_prompt(instruction: str, ground_answer: str, response: str) -> str:
    return f"""
You are judging whether a model successfully completed a correction task.

Task description:
The user instruction contains an incorrect statement about the video.
The model should identify the incorrect part and provide the correct information.

Input:
- User instruction: {instruction}
- Reference correction: {ground_answer}
- Model response: "{response}"

Success criteria:
1. Identify the error implied by the user instruction and the reference correction.
2. Check whether the model corrected all required error points.
3. The corrected content must preserve the correct context, including subject, object, action, and attributes.
4. Ignore extra information that is unrelated to both the instruction and the reference, unless it contradicts them.

Scoring:
- 1 = all required error points are corrected and the context is consistent.
- 0 = at least one required correction is missing, wrong, or contextually inconsistent.

Return exactly this JSON object:
{{
  "success_score": <0 or 1>,
  "reasoning": "<brief explanation covering the required correction points>"
}}
"""


def parse_judge_response(response: str) -> Dict[str, Any]:
    for match in re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", response, flags=re.DOTALL):
        try:
            parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", match))
            parsed["success_score"] = int(float(parsed.get("success_score", 0)))
            return parsed
        except Exception:
            continue

    score_match = re.search(r'"success_score"\s*:\s*([01])', response)
    return {
        "success_score": int(score_match.group(1)) if score_match else 0,
        "reasoning": "",
    }


def llm_judge(
    client: OpenAILLMClient,
    model_id: str,
    task_type: str,
    instruction: str,
    ground_answer: str,
    response: str,
) -> Dict[str, Any]:
    if task_type == "correction":
        prompt = build_correction_prompt(instruction, ground_answer, response)
    else:
        prompt = build_reminder_prompt(instruction, response, task_type)

    api_response = client.chat_completion(
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
        model=model_id,
        temperature=0.1,
        max_tokens=1200,
    )

    if "error" in api_response:
        return {
            "success_score": 0,
            "reasoning": f"API call failed: {api_response.get('error')}",
            "api_response": api_response,
        }

    response_text = extract_message_text(api_response)
    parsed = parse_judge_response(response_text)
    return {
        "success_score": int(parsed.get("success_score", 0)),
        "reasoning": parsed.get("reasoning", ""),
        "api_response": response_text,
    }


def evaluate_sample(
    client: OpenAILLMClient,
    model_id: str,
    instruction: str,
    gt_data: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    task_type: str,
    window_size: float,
) -> Dict[str, Any]:
    reminders = collect_reminders(gt_data)

    if task_type == "correction" or not reminders:
        response = full_response_text(chunks)
        judge = llm_judge(
            client=client,
            model_id=model_id,
            task_type=task_type,
            instruction=instruction,
            ground_answer=str(gt_data.get("answer", "")),
            response=response,
        )
        return {
            "total_score": int(judge["success_score"]),
            "event_scores": [int(judge["success_score"])],
            "event_details": [
                {
                    "reminder_index": None,
                    "start_time": None,
                    "response_segment": response,
                    "score": int(judge["success_score"]),
                    "reasoning": judge.get("reasoning", ""),
                    "judge_result": judge,
                }
            ],
            "all_success": int(judge["success_score"]) == 1,
        }

    event_scores: List[int] = []
    event_details: List[Dict[str, Any]] = []
    for reminder in reminders:
        start_time = float(reminder["start"])
        response_segment = response_text_near_time(chunks, start_time, window_size)
        if not response_segment:
            event_scores.append(0)
            event_details.append(
                {
                    "reminder_index": int(reminder["index"]),
                    "start_time": start_time,
                    "response_segment": "",
                    "score": 0,
                    "reasoning": "No response text was found in the evaluation window.",
                    "judge_result": None,
                }
            )
            continue

        judge = llm_judge(
            client=client,
            model_id=model_id,
            task_type=task_type,
            instruction=instruction,
            ground_answer=str(gt_data.get("answer", "")),
            response=response_segment,
        )
        score = int(judge["success_score"])
        event_scores.append(score)
        event_details.append(
            {
                "reminder_index": int(reminder["index"]),
                "start_time": start_time,
                "response_segment": response_segment,
                "score": score,
                "reasoning": judge.get("reasoning", ""),
                "judge_result": judge,
            }
        )

    all_success = all(score == 1 for score in event_scores)
    return {
        "total_score": 1 if all_success else 0,
        "event_scores": event_scores,
        "event_details": event_details,
        "all_success": all_success,
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate DuplexEval proactive reminder and correction outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--instruction", required=True, help="User instruction text.")
    parser.add_argument("--response", required=True, help="Model response JSON file.")
    parser.add_argument("--task-type", required=True, help="correction, proactive_reminder, or post_event_reminder.")
    parser.add_argument("--output", required=True, help="Output JSON path.")
    parser.add_argument("--gt", default=None, help="Optional ground-truth JSON file.")
    parser.add_argument("--ground-answer", default="", help="Reference correction text for correction tasks.")
    parser.add_argument(
        "--reminder-times",
        nargs="*",
        default=None,
        help="Reminder start times in seconds. Usually from reminder1 and reminder2.",
    )
    parser.add_argument(
        "--model-id",
        default=os.environ.get("DUPLEXEVAL_MODEL", "EVALUATOR_MODEL"),
        help="Evaluator model id. Defaults to DUPLEXEVAL_MODEL when set.",
    )
    parser.add_argument("--window-size", type=float, default=10.0, help="Evaluation window after each reminder time.")
    return parser.parse_args()


def main() -> int:
    args = parse_arguments()
    start_time = time.time()

    try:
        task_type = canonical_task_type(args.task_type)
        response_raw = read_json(args.response)
        chunks = normalize_chunks(response_raw)
        gt_data = load_ground_truth(args)

        print("=" * 72)
        print("DuplexEval Proactive Reminder Evaluation")
        print("=" * 72)
        print(f"Task type: {task_type}")
        print(f"Response file: {args.response}")
        print(f"Output file: {args.output}")
        print(f"Window size: {args.window_size:.2f}s")
        print(f"Response segments: {len(chunks)}")

        client = OpenAILLMClient()
        result = evaluate_sample(
            client=client,
            model_id=args.model_id,
            instruction=args.instruction,
            gt_data=gt_data,
            chunks=chunks,
            task_type=task_type,
            window_size=args.window_size,
        )
        elapsed = round(time.time() - start_time, 2)

        output_data = {
            "metadata": {
                "script": "proactive_reminder.py",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "evaluation_time_seconds": elapsed,
                "instruction": args.instruction,
                "response_file": args.response,
                "gt_file": args.gt,
                "task_type": task_type,
                "model_id": args.model_id,
                "window_size": args.window_size,
            },
            "ground_truth": gt_data,
            "model_response": {
                "file": args.response,
                "num_chunks": len(chunks),
                **response_meta(response_raw),
            },
            "evaluation_result": result,
            "summary": {
                "total_score": result["total_score"],
                "num_events": len(result["event_details"]),
                "successful_events": sum(result["event_scores"]),
                "all_success": result["all_success"],
            },
        }

        write_json(output_data, args.output)
        print(f"Saved results to {args.output}")
        print(f"Total score: {result['total_score']}")
        print(f"Event scores: {result['event_scores']}")
        print(f"Total time: {elapsed:.2f}s")
        return 0

    except Exception as exc:
        print(f"Evaluation failed: {exc}")
        traceback.print_exc()
        try:
            write_json(
                {
                    "metadata": {
                        "script": "proactive_reminder.py",
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                        "args": vars(args),
                    },
                    "error": {
                        "type": type(exc).__name__,
                        "message": str(exc),
                        "traceback": traceback.format_exc(),
                    },
                },
                args.output,
            )
        except Exception:
            pass
        return 1


if __name__ == "__main__":
    sys.exit(main())
