#!/usr/bin/env python3
"""Small HTTP server that runs Alpamayo in a separate Python 3.12 process."""

from __future__ import annotations

import argparse
import base64
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import sys
import traceback

from PIL import Image
import torch


def _strtobool(value: str | None) -> bool:
  return str(value).lower() in ("1", "true", "t", "yes", "y", "on")


def _ensure_alpamayo_on_path(repo: str | None, src: str | None) -> None:
  candidates = []
  if src:
    candidates.append(Path(src))
  if repo:
    candidates.append(Path(repo) / "src")
  candidates.append(Path(__file__).resolve().parents[2] / "alpamayo" / "src")
  for candidate in candidates:
    if candidate.exists():
      candidate_str = str(candidate)
      if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
      return


class AlpamayoVLM:
  def __init__(self, args):
    _ensure_alpamayo_on_path(args.alpamayo_repo, args.alpamayo_src)

    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, ExpertLogitsProcessor
    from alpamayo_r1.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token

    self.helper = helper
    self.expert_logits_processor_cls = ExpertLogitsProcessor
    self.stop_after_eos_cls = StopAfterEOS
    self.replace_padding_after_eos = replace_padding_after_eos
    self.to_special_token = to_special_token
    self.device = args.device
    self.max_new_tokens = args.max_new_tokens
    self.history_steps = args.history_steps
    self.do_sample = args.do_sample

    dtype = self._resolve_dtype(args.dtype)
    self.model = AlpamayoR1.from_pretrained(args.model, dtype=dtype)
    self.model.to(args.device if args.device != "cpu" else "cpu")
    self.model.eval()
    self.processor = helper.get_processor(self.model.tokenizer)
    self.tokenizer = self.model.tokenizer

  def generate(self, image: Image.Image, prompt: str, max_new_tokens: int | None = None) -> dict:
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList

    messages = self._messages(image, prompt)
    inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = self.helper.to_device(inputs, self.device)
    input_ids = inputs.pop("input_ids")
    ego_history_xyz, ego_history_rot = self._stationary_history()
    input_ids = self.model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": ego_history_xyz.to(self.device),
            "ego_history_rot": ego_history_rot.to(self.device),
        })

    eos_token_id = self.tokenizer.convert_tokens_to_ids(self.to_special_token("traj_future_start"))
    generation_config = self.model.vlm.generation_config
    generation_config.do_sample = self.do_sample
    generation_config.max_new_tokens = int(max_new_tokens or self.max_new_tokens)
    generation_config.num_return_sequences = 1
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = self.tokenizer.pad_token_id
    logits_processor = LogitsProcessorList([
        self.expert_logits_processor_cls(
            traj_token_offset=self.model.config.traj_token_start_idx,
            traj_vocab_size=self.model.config.traj_vocab_size,
        )
    ])

    with torch.inference_mode():
      outputs = self.model.vlm.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          stopping_criteria=StoppingCriteriaList([self.stop_after_eos_cls(eos_token_id=eos_token_id)]),
          logits_processor=logits_processor,
          **inputs)

    sequences = self.replace_padding_after_eos(
        token_ids=outputs.sequences,
        eos_token_id=eos_token_id,
        pad_token_id=self.tokenizer.pad_token_id)
    generated_ids = sequences[:, input_ids.shape[1]:]
    raw_response = self.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False)[0]
    decision = self._parse_action(raw_response)
    decision["raw_response"] = raw_response
    return decision

  def _messages(self, image: Image.Image, prompt: str):
    hist_traj_placeholder = "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
    text = (
        f"{hist_traj_placeholder}{prompt}\n"
        "Return final answer as strict JSON only: "
        "{\"action\":\"none|brake|left|right\",\"reason\":\"...\"}."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You choose safe activation-steering interventions."}],
        },
        {
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": text}],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]

  def _stationary_history(self):
    ego_history_xyz = torch.zeros((1, 1, self.history_steps, 3), dtype=torch.float32)
    eye = torch.eye(3, dtype=torch.float32)
    ego_history_rot = eye.reshape(1, 1, 1, 3, 3).repeat(1, 1, self.history_steps, 1, 1)
    return ego_history_xyz, ego_history_rot

  @staticmethod
  def _parse_action(response: str) -> dict:
    data = None
    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if match is not None:
      try:
        data = json.loads(match.group(0))
      except json.JSONDecodeError:
        data = None
    if isinstance(data, dict):
      action = AlpamayoVLM._normalize_action(data.get("action", data.get("cmd", "none")))
      return {
          "action": action,
          "confidence": data.get("confidence", 1.0),
          "reason": str(data.get("reason", data.get("reasoning", "")))[:240],
      }

    lowered = response.lower()
    if re.search(r"\b(change|move|merge)\s+(to\s+)?(the\s+)?left\b", lowered):
      return {"action": "left", "confidence": 1.0, "reason": response.strip()[:240]}
    if re.search(r"\b(change|move|merge)\s+(to\s+)?(the\s+)?right\b", lowered):
      return {"action": "right", "confidence": 1.0, "reason": response.strip()[:240]}
    if re.search(r"\b(stop|brake|yield|emergency brake)\b", lowered):
      return {"action": "brake", "confidence": 1.0, "reason": response.strip()[:240]}
    return {"action": "none", "confidence": 1.0, "reason": response.strip()[:240]}

  @staticmethod
  def _normalize_action(action) -> str:
    normalized = re.sub(r"[^a-z_]+", "", str(action).lower())
    if normalized in ("stop", "brake", "yield", "emergency_brake", "emergencybrake"):
      return "brake"
    if normalized in ("left", "l_change", "lchange", "left_change", "leftchange", "change_lane_left", "changelaneleft"):
      return "left"
    if normalized in ("right", "r_change", "rchange", "right_change", "rightchange", "change_lane_right", "changelaneright"):
      return "right"
    return "none"

  @staticmethod
  def _resolve_dtype(dtype_name: str):
    if dtype_name in ("float16", "fp16"):
      return torch.float16
    if dtype_name in ("bfloat16", "bf16", "auto"):
      return torch.bfloat16
    if dtype_name in ("float32", "fp32"):
      return torch.float32
    return torch.bfloat16


class Handler(BaseHTTPRequestHandler):
  model: AlpamayoVLM | None = None

  def do_GET(self):  # pylint: disable=invalid-name
    if self.path != "/health":
      self.send_error(404)
      return
    self._write_json({"ok": True})

  def do_POST(self):  # pylint: disable=invalid-name
    if self.path != "/generate":
      self.send_error(404)
      return
    try:
      length = int(self.headers.get("Content-Length", "0"))
      payload = json.loads(self.rfile.read(length).decode("utf-8"))
      image_bytes = base64.b64decode(payload["image_base64"])
      image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
      result = self.model.generate(
          image=image,
          prompt=str(payload.get("prompt", "")),
          max_new_tokens=payload.get("max_new_tokens"))
      result["frame_id"] = payload.get("frame_id")
      self._write_json(result)
    except Exception as exc:  # pylint: disable=broad-except
      self._write_json(
          {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
          status=500)

  def log_message(self, fmt, *args):
    if _strtobool(os.environ.get("ALPAMAYO_SERVER_VERBOSE", "0")):
      super().log_message(fmt, *args)

  def _write_json(self, payload: dict, status: int = 200):
    body = json.dumps(payload).encode("utf-8")
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.send_header("Content-Length", str(len(body)))
    self.end_headers()
    self.wfile.write(body)


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--host", default=os.environ.get("ALPAMAYO_SERVER_HOST", "127.0.0.1"))
  parser.add_argument("--port", type=int, default=int(os.environ.get("ALPAMAYO_SERVER_PORT", 8765)))
  parser.add_argument("--model", default=os.environ.get("ALPAMAYO_MODEL", "nvidia/Alpamayo-R1-10B"))
  parser.add_argument("--device", default=os.environ.get("ALPAMAYO_DEVICE", "cuda:0"))
  parser.add_argument("--dtype", default=os.environ.get("ALPAMAYO_DTYPE", "bfloat16"))
  parser.add_argument("--max-new-tokens", type=int, default=int(os.environ.get("ALPAMAYO_MAX_NEW_TOKENS", 128)))
  parser.add_argument("--history-steps", type=int, default=int(os.environ.get("ALPAMAYO_HISTORY_STEPS", 16)))
  parser.add_argument("--alpamayo-repo", default=os.environ.get("ALPAMAYO_REPO"))
  parser.add_argument("--alpamayo-src", default=os.environ.get("ALPAMAYO_SRC"))
  parser.add_argument("--do-sample", action="store_true", default=_strtobool(os.environ.get("ALPAMAYO_DO_SAMPLE", "0")))
  return parser.parse_args()


def main():
  args = parse_args()
  Handler.model = AlpamayoVLM(args)
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"Alpamayo VLM server listening on http://{args.host}:{args.port}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
