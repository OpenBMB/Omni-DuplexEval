# Omni-DuplexEval: Evaluating Real-time Duplex Omni-modal Interaction

<p align="center">
  📄 <a href="https://arxiv.org/abs/2605.17360" target="_blank">Paper</a> &nbsp; | &nbsp;
  🤗 <a href="https://huggingface.co/datasets/Hothan/Omni-DuplexEval" target="_blank">Hugging Face Dataset</a> &nbsp;
</p>

This directory contains the evaluation scripts for the Omni-DuplexEval benchmark.

## Tasks

Omni-DuplexEval contains two evaluation families:

- `real_time_description.py`: evaluates real-time description outputs with temporal sensitivity and content accuracy.
- `proactive_reminder.py`: evaluates proactive event reminders, post-event reminders, and correction-style proactive responses.

Batch wrappers are provided for the HuggingFace dataset format:

- `batch_real_time_description.py`
- `batch_proactive_reminder.py`

## Environment

Tested environment:

- Python 3.10+
- Linux
- `ffmpeg` and `ffprobe` available on `PATH`
- Python dependencies in `requirements.txt`
- An OpenAI-compatible chat-completions endpoint that supports image and video message content

Install dependencies:

```bash
cd /path/to/Omni-DuplexEval
python -m pip install -r requirements.txt
```

Install system video tools if needed:

```bash
sudo apt-get update
sudo apt-get install -y ffmpeg
```

Configure the evaluator API:

```bash
export DUPLEXEVAL_API_KEY="YOUR_API_KEY"
export DUPLEXEVAL_BASE_URL="YOUR_OPENAI_COMPATIBLE_BASE_URL"
export DUPLEXEVAL_MODEL="YOUR_EVALUATOR_MODEL_ID"
```

`OPENAI_API_KEY` and `OPENAI_BASE_URL` are also accepted.

## Dataset

The dataset uses the following unified fields:

- `id`: sample id.
- `question_text`: instruction text.
- `answer1`: correction reference or first real-time-description reference.
- `answer2`: second real-time-description reference, when available.
- `reminder1`: first reminder timestamp in seconds, when available.
- `reminder2`: second reminder timestamp in seconds, when available.
- `video_type`: source video type.
- `video_duration`: video duration in seconds.
- `video`: video file.
- `question_audio`: instruction audio.

The batch scripts read the dataset directly through `datasets.load_dataset`. Intermediate video files may be materialized under the output directory while evaluation is running.

## Model Response Format

The evaluation scripts assume that model or baseline responses have already been generated. A response JSON file may be either a top-level list or an object containing `chunks` or `sentences`.

Accepted segment fields:

```json
[
  {"sentence": "The person opens the door.", "start": 4.2, "end": 5.1},
  {"text": "Now the person walks inside.", "current_time": 6.0}
]
```

For batch evaluation, place responses by default at:

```text
{response_root}/{split}/{id}.json
```

or pass a custom template:

```bash
--response-template "{response_root}/my_model/{split}/{id}.json"
```

## Single-Sample Commands

Real-time description:

```bash
python real_time_description.py \
  --video /path/to/video.mp4 \
  --input /path/to/model_response.json \
  --question "Describe what happens in the video in real time." \
  --gt-text "Reference description 1" "Reference description 2" \
  --output /path/to/output/real_time_description_result.json \
  --model "$DUPLEXEVAL_MODEL" \
  --metrics all \
  --fps 2 \
  --max-workers 8
```

Proactive reminder:

```bash
python proactive_reminder.py \
  --instruction "Remind me when the person picks up the cup." \
  --response /path/to/model_response.json \
  --task-type proactive_reminder \
  --reminder-times 15.0 \
  --output /path/to/output/proactive_reminder_result.json \
  --model-id "$DUPLEXEVAL_MODEL" \
  --window-size 10.0
```

Correction:

```bash
python proactive_reminder.py \
  --instruction "The exterior of this cruise ship sailing in the sea is black." \
  --response /path/to/model_response.json \
  --task-type correction \
  --ground-answer "The exterior of this cruise ship sailing in the sea is white." \
  --output /path/to/output/correction_result.json \
  --model-id "$DUPLEXEVAL_MODEL"
```

## Batch Commands

Real-time description splits:

```bash
python batch_real_time_description.py \
  --dataset foragi/Omni-DuplexEval-Examples \
  --response-root /path/to/model_responses \
  --output-root /path/to/eval_outputs/real_time_description \
  --model "$DUPLEXEVAL_MODEL" \
  --metrics all \
  --fps 2 \
  --eval-workers 8 \
  --sample-workers 2 \
  --overwrite
```

Proactive reminder and correction splits:

```bash
python batch_proactive_reminder.py \
  --dataset foragi/Omni-DuplexEval-Examples \
  --response-root /path/to/model_responses \
  --output-root /path/to/eval_outputs/proactive_reminder \
  --model-id "$DUPLEXEVAL_MODEL" \
  --window-size 10.0 \
  --sample-workers 4 \
  --overwrite
```

Dry-run mode prints the planned work without API calls:

```bash
python batch_real_time_description.py \
  --dataset foragi/Omni-DuplexEval-Examples \
  --response-root /path/to/model_responses \
  --output-root /tmp/Omni-DuplexEval_dryrun \
  --dry-run \
  --limit 1
```

## Included and Not Included

Included:

- Evaluation prompts and scoring logic for the two submitted evaluation families.
- Single-sample scripts.
- Batch scripts for all HuggingFace dataset splits.
- Dataset access instructions and exact commands.

Not included:

- Model inference or baseline generation code. These scripts evaluate already generated response JSON files. In our paper, model inference was performed with each model's open-source codebase. To reproduce a method or baseline end-to-end, first generate responses in the JSON format above, then run the batch commands.

## Output

Each sample writes a JSON file with metadata, inputs, detailed per-event or per-sentence judgments, summary scores, and elapsed time.

Batch scripts additionally write:

- `batch_real_time_description_summary.json`
- `batch_proactive_reminder_summary.json`

## Citation

**BibTeX:**
```bibtex
@misc{he2026omniduplexevalevaluatingrealtimeduplex,
      title={Omni-DuplexEval: Evaluating Real-time Duplex Omni-modal Interaction}, 
      author={Chaoqun He and Mingyang Xiang and Yingjing Xu and Bokai Xu and Junbo Cui and Jie Zhou and Yuan Yao and Lijie Wen},
      year={2026},
      eprint={2605.17360},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2605.17360}, 
}
```

