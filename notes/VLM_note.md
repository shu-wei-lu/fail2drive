# VLM Steering Gate Notes

This note documents how to run the asynchronous VLM steering gate added to
`fail2drive/team_code`.

## What It Does

The driving loop still runs synchronously through `SensorAgent.run_step()`.
The VLM runs in a background thread and never blocks CARLA ticks.

At each tick:

1. `SensorAgent` reads the latest completed VLM decision.
2. The decision's `steering_alpha` is passed to `LidarCenterNet.forward(..., steering_alpha=...)`.
3. The TransFuser model predicts trajectory/checkpoints.
4. The predicted trajectory is projected onto the front RGB image.
5. The overlaid image is submitted to the VLM worker.
6. The VLM worker processes whichever latest frame is available after it finishes the previous one.

So if the VLM starts on frame 1 and finishes around frame 10, it will then pick
the newest queued frame, not every intermediate frame.

## Files

- `fail2drive/team_code/vlm_gate.py`
  - `AsyncVLMGate`
  - HuggingFace/Qwen inference
  - background thread and latest-frame queue
- `fail2drive/team_code/sensor_agent.py`
  - starts/stops the VLM worker
  - overlays predicted trajectory on front image
  - submits images to the VLM worker
- `fail2drive/team_code/model.py`
  - accepts `steering_alpha`
  - skips TTC detection when VLM supplies an alpha
  - mixes steering features using the VLM-provided alpha

## Dependencies

The VLM path needs a recent `transformers` version, `accelerate`, and
`bitsandbytes` for 4-bit/8-bit loading.

`fail2drive/team_code/requirements.txt` now includes:

```txt
accelerate>=0.26.0
bitsandbytes>=0.43.0
transformers>=4.49.0
```

Install/update dependencies before enabling VLM:

```bash
pip install -r fail2drive/team_code/requirements.txt
```

If using Qwen2.5-VL, you may also need the model-specific optional dependencies
required by your installed `transformers` version.

## Minimal VLM Run

Start CARLA first, then from the repo root:

```bash
export VLM_STEERING=1
export VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct
export VLM_DEVICE=cuda:1
export VLM_QUANTIZATION=4bit
export VLM_CONF_THRESHOLD=0.6

python -u fail2drive/local_evaluate.py \
  --routes fail2drive_split \
  --out_root results/fail2drive_vlm \
  --seeds 1 \
  --route-limit 1 \
  --agent_file fail2drive/team_code/sensor_agent.py \
  --agent_config /path/to/your/checkpoint_dir \
  --lb_script fail2drive/leaderboard/leaderboard/leaderboard_evaluator_local.py \
  --host localhost \
  --port 2000
```

`--agent_config` must point to the directory containing `config.json` and
`model*.pth`.

## Download/Test Qwen First

To download the Qwen-VL model into the HuggingFace cache and run one simple
image inference:

```bash
python -u fail2drive/tools/qwen_vl_inference_example.py \
  --model Qwen/Qwen2.5-VL-7B-Instruct \
  --image /path/to/front_image.png \
  --device cuda:1 \
  --quantization 4bit
```

This should print a small JSON-like response with `steering_alpha`,
`confidence`, and `reason`. The same model cache will be reused by
`AsyncVLMGate`.

## Useful Environment Variables

```bash
# Enable/disable the VLM gate. Default: False
export VLM_STEERING=1

# HuggingFace model id. Default: Qwen/Qwen2.5-VL-7B-Instruct
export VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct

# Device for VLM. Default: cuda:0
export VLM_DEVICE=cuda:1

# Optional quantization: none, 4bit, or 8bit. Default: none
export VLM_QUANTIZATION=4bit

# Equivalent compatibility flags. VLM_QUANTIZATION is preferred.
export VLM_LOAD_IN_4BIT=1
export VLM_LOAD_IN_8BIT=0

# Optional dtype. Default: auto
export VLM_TORCH_DTYPE=auto

# Confidence threshold for accepting steering_alpha > 0. Default: 0.6
export VLM_CONF_THRESHOLD=0.6

# Try submitting one latest frame every N agent frames. Default: 1
export VLM_EVERY_N=1

# Max generated tokens for the JSON response. Default: 96
export VLM_MAX_NEW_TOKENS=96

# Print VLM submit/result/use logs. Default: 1
export VLM_VERBOSE=1

# Save each image actually processed by the VLM worker. Default: 0
export VLM_SAVE_INPUTS=1

# Directory for saved VLM input images. Default: vlm_inputs
export VLM_INPUT_SAVE_DIR=results/vlm_inputs

# Number of points drawn on the VLM trajectory overlay. Default: 10
export VLM_TRAJECTORY_POINTS=10

# Time horizon used to resample the VLM trajectory overlay. Default: 2.0 seconds
export VLM_TRAJECTORY_HORIZON_S=2.0

# Z height used when projecting ego-frame predicted trajectory to the image.
# Default: 0.0 meters.
export VLM_TRAJECTORY_Z_M=0.0
```

## Custom Prompt

You can override the default prompt:

```bash
export VLM_PROMPT='Decide how strongly to apply steering intervention based on whether the orange predicted ego trajectory intersects an immediate obstacle or blocked lane. Return only JSON: {"steering_alpha": number, "confidence": number, "reason": string}.'
```

If the prompt contains `{speed_m_s}`, `{command}`, or `{target_point}`, they are
formatted with runtime values. If formatting fails, the prompt is used as plain
text and context is appended.

## Expected VLM Output

The gate expects JSON like:

```json
{
  "steering_alpha": 0.75,
  "confidence": 0.82,
  "reason": "Vehicle is blocking the predicted path ahead."
}
```

`steering_alpha` is clamped to `[0, 1]`. If confidence is below
`VLM_CONF_THRESHOLD`, the gate forces `steering_alpha=0`.

## Steering Logic

The VLM does not directly control steer, throttle, or brake.

It controls the feature-mixing strength for steering feature injection:

```text
target_speed_features =
  (1 - steering_alpha) * target_speed_features
  + steering_alpha * steering_feats
```

When `VLM_STEERING=1`, `SensorAgent` always passes a numeric alpha, so
`model.py` skips the TTC region detector. With `VLM_STEERING=0`, the model keeps
the old TTC-based steering behavior.

## Debug Tips

- Keep `VLM_STEERING=0` for baseline runs.
- Start with `--route-limit 1` for smoke tests.
- If the VLM fails to initialize, the worker stores a disabled decision and
  prints the traceback; the CARLA tick loop should keep running.
- If GPU memory is tight, use a smaller Qwen-VL model or run VLM on another
  device with `VLM_DEVICE`.
