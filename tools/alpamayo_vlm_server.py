#!/usr/bin/env python3
"""Small HTTP server that runs Alpamayo in a separate Python 3.12 process."""

from __future__ import annotations

import argparse
import base64
import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import os
from pathlib import Path
import re
import sys
import threading
import time
import traceback

import numpy as np
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
  candidates.append(Path(__file__).resolve().parents[2] / "alpamayo1.5" / "src")
  for candidate in candidates:
    if candidate.exists():
      candidate_str = str(candidate)
      if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
      return


class AlpamayoVLM:
  def __init__(self, args):
    _ensure_alpamayo_on_path(args.alpamayo_repo, args.alpamayo_src)

    self.version = self._resolve_version(args.alpamayo_version, args.model)
    self.device = args.device
    self.max_new_tokens = args.max_new_tokens
    self.history_steps = args.history_steps
    self.do_sample = args.do_sample
    self.vqa_top_p = args.vqa_top_p
    self.vqa_temperature = args.vqa_temperature
    self.vqa_num_samples = args.vqa_num_samples
    self.raw_output_only = args.raw_output_only

    dtype = self._resolve_dtype(args.dtype)
    if self.version == "1.5":
      from alpamayo1_5 import helper
      from alpamayo1_5.models.alpamayo1_5 import Alpamayo1_5

      model_kwargs = {"dtype": dtype}
      if args.attn_implementation:
        model_kwargs["attn_implementation"] = args.attn_implementation
      self.helper = helper
      self.model = Alpamayo1_5.from_pretrained(args.model, **model_kwargs)
      self.model.to(args.device if args.device != "cpu" else "cpu")
      self.model.eval()
      self.processor = helper.get_processor(self.model.tokenizer)
      self.tokenizer = self.model.tokenizer
      print(f"Loaded Alpamayo 1.5 VQA backend model={args.model}", flush=True)
    else:
      from alpamayo_r1 import helper
      from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, ExpertLogitsProcessor
      from alpamayo_r1.models.token_utils import StopAfterEOS, replace_padding_after_eos, to_special_token

      self.helper = helper
      self.expert_logits_processor_cls = ExpertLogitsProcessor
      self.stop_after_eos_cls = StopAfterEOS
      self.replace_padding_after_eos = replace_padding_after_eos
      self.to_special_token = to_special_token
      self.model = AlpamayoR1.from_pretrained(args.model, dtype=dtype)
      self.model.to(args.device if args.device != "cpu" else "cpu")
      self.model.eval()
      self.processor = helper.get_processor(self.model.tokenizer)
      self.tokenizer = self.model.tokenizer
      print(f"Loaded Alpamayo 1 backend model={args.model}", flush=True)

  def generate(self,
               images: list[Image.Image],
               prompt: str,
               max_new_tokens: int | None = None,
               ego_history_xyz=None,
               ego_history_rot=None,
               speed=None,
               command=None,
               target_point=None,
               image_frame_ids=None) -> dict:
    if self.version == "1.5":
      return self._generate_v15(
          images=images,
          max_new_tokens=max_new_tokens,
          speed=speed,
          command=command,
          target_point=target_point,
          image_frame_ids=image_frame_ids)

    return self._generate_v1(
        images=images,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        ego_history_xyz=ego_history_xyz,
        ego_history_rot=ego_history_rot)

  def _generate_v1(self,
                   images: list[Image.Image],
                   prompt: str,
                   max_new_tokens: int | None = None,
                   ego_history_xyz=None,
                   ego_history_rot=None) -> dict:
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList

    messages = self._messages(images, prompt)
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
    ego_history_xyz, ego_history_rot = self._ego_history_or_stationary(
        ego_history_xyz,
        ego_history_rot)
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
        token_ids=outputs.sequences.clone(),
        eos_token_id=eos_token_id,
        pad_token_id=self.tokenizer.pad_token_id)
    generated_ids = sequences[:, input_ids.shape[1]:]
    raw_response = self.tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False)[0]
    decision = self._parse_action(raw_response, require_explicit=self.version == "1.5")
    decision["raw_response"] = raw_response
    return decision

  def _generate_v15(self,
                    images: list[Image.Image],
                    max_new_tokens: int | None = None,
                    speed=None,
                    command=None,
                    target_point=None,
                    image_frame_ids=None) -> dict:
    frames = torch.stack([self._pil_to_chw_tensor(image) for image in images], dim=0)
    camera_indices = torch.tensor([1], dtype=torch.int64)
    question = self._v15_question(
        num_images=len(images),
        speed=speed,
        command=command,
        target_point=target_point,
        image_frame_ids=image_frame_ids)
    messages = self.helper.create_vqa_message(
        frames,
        question=question,
        camera_indices=camera_indices,
        num_frames_per_camera=len(images))
    inputs = self.processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    model_inputs = {"tokenized_data": inputs}
    model_inputs = self.helper.to_device(model_inputs, self.device)

    with torch.inference_mode():
      if self.device != "cpu" and str(self.device).startswith("cuda"):
        autocast_ctx = torch.autocast("cuda", dtype=self._resolve_dtype("bfloat16"))
      else:
        autocast_ctx = contextlib.nullcontext()
      with autocast_ctx:
        extra = self.model.generate_text(
            data=model_inputs,
            top_p=self.vqa_top_p,
            temperature=self.vqa_temperature,
            num_samples=self.vqa_num_samples,
            max_generation_length=int(max_new_tokens or self.max_new_tokens))

    raw_response = self._first_nonempty_text(extra, ("answer", "meta_action", "cot"))
    if self.raw_output_only:
      decision = {
          "action": "none",
          "confidence": 0.0,
          "reason": raw_response[:240],
      }
    else:
      decision = self._parse_v15_answer(raw_response)
    decision["question"] = question
    decision["raw_response"] = raw_response
    decision["raw_generate_text_output"] = self._jsonable(extra)
    return decision

  def _messages(self, images: list[Image.Image], prompt: str):
    hist_traj_placeholder = "<|traj_history_start|>" + "<|traj_history|>" * 48 + "<|traj_history_end|>"
    frame_context = (
        "Images are ordered from oldest front camera frame to current front camera frame."
        if len(images) > 1 else
        "Image is the current front camera frame."
    )
    text = (
        f"{hist_traj_placeholder}{prompt}\n"
        f"{frame_context}\n"
        "Return final answer as strict JSON only: "
        "{\"action\":\"none|brake|left|right\",\"reason\":\"...\"}."
    )
    return [
        {
            "role": "system",
            "content": [{"type": "text", "text": "You are a driving assistant that generates safe and accurate actions."}],
        },
        {
            "role": "user",
            "content": (
                [{"type": "image", "image": image} for image in images]
                + [{"type": "text", "text": text}]
            ),
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]

  @staticmethod
  def _pil_to_chw_tensor(image: Image.Image) -> torch.Tensor:
    array = np.asarray(image.convert("RGB"))
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()

  @staticmethod
  def _v15_question(num_images: int, speed=None, command=None, target_point=None, image_frame_ids=None) -> str:
    frame_context = (
        "The images are ordered from oldest front camera frame to current front camera frame."
        if num_images > 1 else
        "The image is the current front camera frame."
    )
    speed_text = "unknown" if speed is None else f"{float(speed):.2f} m/s"
    command_text = AlpamayoVLM._navigation_intent(command)
    target_text = AlpamayoVLM._format_target_point(target_point)
    frame_text = ""
    if image_frame_ids:
      frame_text = f" Image frame ids: {image_frame_ids}."
    # return (
    #     "What are the key traffic elements in the front camera frame(s), and what immediate driving "
    #     "behavior should the ego vehicle take: continue normally, slow down or stop, steer left, or steer right?\n"
    #     f"{frame_context}\n"
    #     f"Context: ego speed is {speed_text}; navigation intent is {command_text}; "
    #     f"ego-frame target point is {target_text}.{frame_text}\n"
    #     "Explain briefly. If there is no immediate hazard in the ego path, say the vehicle should continue normally. "
    #     "If the ego path is blocked or uncertain, say it should slow down or stop. "
    #     "Only recommend steering left or right when that side clearly has safer free space."
    # )
    # return (
    #     "Analyze the immediate driving scene. "
    #     f"{frame_context}\n"
    #     f"Context: ego speed is {speed_text}; navigation intent is {command_text}; "
    #     f"ego-frame target point is {target_text}.{frame_text}\n"
    #     "Task 1: Briefly state the single most critical object or free space in the ego path.\n"
    #     "Task 2: Based on Task 1, you MUST choose EXACTLY ONE emergency action from: [continue normally, brake, steer left, steer right].\n"
    #     "Give only ONE definitive reasoning and ONE action."
    # )

#     return (
#         "Do you see any hazard in current scene? Should the driver brake, steer left or steer right?"
# )
    return (
      "Identify potential hazards in the scene. If there are any, give one action suggestion of the following: brake, steer left, steer right."
    )
  @staticmethod
  def _navigation_intent(command) -> str:
    try:
      command = int(command)
    except (TypeError, ValueError):
      return "unknown"
    return {
        1: "turn left at intersection",
        2: "turn right at intersection",
        3: "go straight",
        4: "follow lane",
        5: "change lane left",
        6: "change lane right",
    }.get(command, "follow lane")

  @staticmethod
  def _format_target_point(target_point) -> str:
    if target_point is None:
      return "unknown"
    try:
      flat = np.asarray(target_point, dtype=np.float32).reshape(-1)
      if flat.size >= 2:
        return f"({float(flat[0]):.2f}, {float(flat[1]):.2f})"
    except (TypeError, ValueError):
      pass
    return str(target_point)

  @staticmethod
  def _first_text(value) -> str:
    if isinstance(value, str):
      return value
    if isinstance(value, np.ndarray):
      if value.size == 0:
        return ""
      return AlpamayoVLM._first_text(value.reshape(-1)[0])
    if hasattr(value, "tolist") and not isinstance(value, (list, tuple, dict)):
      return AlpamayoVLM._first_text(value.tolist())
    if isinstance(value, (list, tuple)):
      if not value:
        return ""
      return AlpamayoVLM._first_text(value[0])
    return str(value)

  @staticmethod
  def _first_nonempty_text(extra: dict, keys: tuple[str, ...]) -> str:
    for key in keys:
      text = AlpamayoVLM._first_text(extra.get(key, "")).strip()
      if text:
        return text
    return ""

  @staticmethod
  def _jsonable(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
      return value
    if isinstance(value, dict):
      return {str(key): AlpamayoVLM._jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
      return [AlpamayoVLM._jsonable(item) for item in value]
    if isinstance(value, np.ndarray):
      return value.tolist()
    if torch.is_tensor(value):
      return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
      try:
        return value.tolist()
      except Exception:
        pass
    return str(value)

  def _stationary_history(self):
    ego_history_xyz = torch.zeros((1, 1, self.history_steps, 3), dtype=torch.float32)
    eye = torch.eye(3, dtype=torch.float32)
    ego_history_rot = eye.reshape(1, 1, 1, 3, 3).repeat(1, 1, self.history_steps, 1, 1)
    return ego_history_xyz, ego_history_rot

  def _ego_history_or_stationary(self, ego_history_xyz, ego_history_rot):
    if ego_history_xyz is None or ego_history_rot is None:
      return self._stationary_history()
    xyz = torch.as_tensor(ego_history_xyz, dtype=torch.float32)
    rot = torch.as_tensor(ego_history_rot, dtype=torch.float32)
    if xyz.ndim != 4 or xyz.shape[-1] != 3:
      raise ValueError(f"ego_history_xyz must have shape [B,N,T,3], got {tuple(xyz.shape)}")
    if rot.ndim != 5 or rot.shape[-2:] != (3, 3):
      raise ValueError(f"ego_history_rot must have shape [B,N,T,3,3], got {tuple(rot.shape)}")
    return xyz, rot

  @staticmethod
  def _parse_action(response: str, require_explicit: bool = False) -> dict:
    response = AlpamayoVLM._clean_generated_text(response)
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

    explicit = re.search(
        r"\b(?:intervention|action|cmd|command)\s*[:=]\s*['\"]?(none|brake|left|right|stop|yield|change lane left|change lane right)\b",
        response,
        flags=re.IGNORECASE)
    if explicit is not None:
      action = AlpamayoVLM._normalize_action(explicit.group(1))
      reason = AlpamayoVLM._extract_reason_text(response)
      return {"action": action, "confidence": 1.0, "reason": reason[:240]}

    if require_explicit:
      reason = AlpamayoVLM._invalid_or_uncertain_reason(response)
      return {"action": "none", "confidence": 0.0, "reason": reason[:240]}

    lowered = response.lower()
    if re.search(r"\b(change|move|merge)\s+(to\s+)?(the\s+)?left\b", lowered):
      return {"action": "left", "confidence": 1.0, "reason": response.strip()[:240]}
    if re.search(r"\b(change|move|merge)\s+(to\s+)?(the\s+)?right\b", lowered):
      return {"action": "right", "confidence": 1.0, "reason": response.strip()[:240]}
    if re.search(r"\b(stop|brake|yield|emergency brake)\b", lowered):
      return {"action": "brake", "confidence": 1.0, "reason": response.strip()[:240]}
    cleaned = AlpamayoVLM._clean_generated_text(response)
    return {
        "action": "none",
        "confidence": 1.0,
        "reason": cleaned[:240] or "No JSON action emitted before trajectory start",
    }

  @staticmethod
  def _extract_reason_text(response: str) -> str:
    match = re.search(
        r"\bcoc\s*[:=]\s*(.*?)(?:\n\s*(?:intervention|action|cmd|command)\s*[:=]|$)",
        response,
        flags=re.IGNORECASE | re.DOTALL)
    if match is not None:
      return AlpamayoVLM._clean_generated_text(match.group(1))
    match = re.search(r"\breason\s*[:=]\s*(.*)", response, flags=re.IGNORECASE | re.DOTALL)
    if match is not None:
      return AlpamayoVLM._clean_generated_text(match.group(1))
    cleaned = AlpamayoVLM._clean_generated_text(response)
    cleaned = re.sub(r"\b(?:intervention|action|cmd|command)\s*[:=]\s*['\"]?(none|brake|left|right|stop|yield)\b", "",
                     cleaned,
                     flags=re.IGNORECASE).strip()
    return cleaned

  @staticmethod
  def _parse_v15_answer(response: str) -> dict:
    response = AlpamayoVLM._clean_generated_text(response)
    explicit = AlpamayoVLM._parse_action(response, require_explicit=True)
    if explicit.get("confidence", 0.0) > 0.0:
      return explicit

    lowered = response.lower()
    if not lowered:
      return {"action": "none", "confidence": 0.0, "reason": "Invalid empty Alpamayo VQA answer"}

    no_hazard_patterns = (
        r"\bno[,;]?\s+i\s+do\s+not\s+see\s+any\s+hazard\b",
        r"\bno[,;]?\s+i\s+don't\s+see\s+any\s+hazard\b",
        r"\bdo\s+not\s+see\s+any\s+hazard\b",
        r"\bdon't\s+see\s+any\s+hazard\b",
        r"\bno\s+(?:immediate\s+)?hazard\b",
        r"\bno\s+(?:obstacles?|dangerous situations?|immediate need)\b",
        r"\broad\s+is\s+clear\b",
        r"\bpath\s+is\s+clear\b",
        r"\blane\s+is\s+clear\b",
        r"\bclear\s+ahead\b",
        r"\bsafe\s+to\s+continue\b",
        r"\bcontinue\s+(?:our|my|the)?\s*(?:current\s+)?driving\b",
        r"\bcontinue\s+(?:our|my|the)?\s*(?:current\s+)?driving\s+behavior\b",
        r"\bcontinue\s+driving\s+at\s+(?:our|my|the)?\s*(?:current\s+)?speed\b",
        r"\bcontinue\s+(?:normally|current action|current behavior)\b",
        r"\bmaintain(?:ing)?\s+(?:a\s+)?safe\s+distance\b",
        r"\bmaintain(?:ing)?\s+(?:their|its|our|my|the)\s+lanes?\b",
        r"\bmaintain(?:ing)?\s+(?:our|my|the)?\s*(?:current\s+)?lane\b",
        r"\bno\s+(?:immediate\s+)?need\s+for\s+(?:braking|steering|braking or steering)\b",
        r"\bno\s+(?:immediate\s+)?need\s+to\s+(?:brake|steer)\b",
    )
    if AlpamayoVLM._has_nonnegated_match(lowered, no_hazard_patterns):
      return {"action": "none", "confidence": 1.0, "reason": response[:240]}

    brake_patterns = (
        r"\b(should|must|need(?:s)? to|best to|important to)\s+(?:slow down|slow|stop|brake|yield|decelerate|reduce speed)\b",
        r"\b(slow down|slow|stop|brake|braking|yield|decelerate|reduce speed|slowly approach)\b",
        r"\b(blocked|blocking|obstructed|obstacle|pedestrian|collision|hazard|unsafe|uncertain)\b",
    )
    left_patterns = (
        r"\b(should|must|need(?:s)? to|best to)\s+(?:steer|move|merge|change lanes?|swerve)\s+(to\s+)?(the\s+)?left\b",
        r"\b(steer|move|merge|change lanes?|swerve)\s+(to\s+)?(the\s+)?left\b",
        r"\bleft\s+(lane|side)\s+(is\s+)?(clear|safer|available)\b",
    )
    right_patterns = (
        r"\b(should|must|need(?:s)? to|best to)\s+(?:steer|move|merge|change lanes?|swerve)\s+(to\s+)?(the\s+)?right\b",
        r"\b(steer|move|merge|change lanes?|swerve)\s+(to\s+)?(the\s+)?right\b",
        r"\bright\s+(lane|side)\s+(is\s+)?(clear|safer|available)\b",
    )
    continue_patterns = (
        r"\b(continue normally|continue|proceed|maintain|maintaining|keep)\b",
        r"\b(no immediate hazard|no hazard|path is clear|lane is clear|clear ahead|no need)\b",
    )

    if AlpamayoVLM._has_nonnegated_match(lowered, left_patterns):
      return {"action": "left", "confidence": 0.8, "reason": response[:240]}
    if AlpamayoVLM._has_nonnegated_match(lowered, right_patterns):
      return {"action": "right", "confidence": 0.8, "reason": response[:240]}
    if AlpamayoVLM._has_nonnegated_match(lowered, brake_patterns):
      return {"action": "brake", "confidence": 0.8, "reason": response[:240]}
    if AlpamayoVLM._has_nonnegated_match(lowered, continue_patterns):
      return {"action": "none", "confidence": 1.0, "reason": response[:240]}

    return {
        "action": "none",
        "confidence": 0.0,
        "reason": AlpamayoVLM._invalid_or_uncertain_reason(response)[:240],
    }

  @staticmethod
  def _has_nonnegated_match(text: str, patterns: tuple[str, ...]) -> bool:
    for pattern in patterns:
      for match in re.finditer(pattern, text, flags=re.IGNORECASE):
        prefix = text[max(0, match.start() - 32):match.start()]
        if re.search(r"\b(no|not|never|avoid|don't|do not|without|no need to)\b", prefix):
          continue
        return True
    return False

  @staticmethod
  def _clean_generated_text(text: str) -> str:
    text = re.sub(r"<\|[^|]+?\|>", "", str(text))
    text = re.sub(r"<i\d+>", "", text)
    return text.strip()

  @staticmethod
  def _invalid_or_uncertain_reason(response: str) -> str:
    cleaned = AlpamayoVLM._clean_generated_text(response)
    if not cleaned:
      return "Invalid empty Alpamayo VQA answer"
    if len(cleaned) < 12:
      return f"Invalid short Alpamayo VQA answer: {cleaned}"
    return f"No explicit action line in Alpamayo VQA answer: {cleaned[:180]}"

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

  @staticmethod
  def _resolve_version(version: str, model_name: str) -> str:
    value = str(version).lower()
    if value in ("1.5", "v1.5", "15", "alpamayo1.5"):
      return "1.5"
    if value in ("1", "v1", "r1", "alpamayo1"):
      return "1"
    lowered_model = str(model_name).lower()
    if "1.5" in lowered_model or "1_5" in lowered_model:
      return "1.5"
    return "1"


class Handler(BaseHTTPRequestHandler):
  model: AlpamayoVLM | None = None
  raw_log_path: Path | None = None
  raw_log_lock = threading.Lock()
  input_save_dir: Path | None = None
  input_save_lock = threading.Lock()
  input_save_counter = 0
  input_save_session = ""

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
      encoded_images = payload.get("images_base64")
      if encoded_images is None:
        encoded_images = [payload["image_base64"]]
      images = [
          Image.open(io.BytesIO(base64.b64decode(encoded_image))).convert("RGB")
          for encoded_image in encoded_images
      ]
      frame_id = payload.get("frame_id")
      image_frame_ids = payload.get("image_frame_ids")
      ego_history_xyz = payload.get("ego_history_xyz")
      ego_history_rot = payload.get("ego_history_rot")
      request_dir = self._save_request_inputs(payload, images, frame_id, image_frame_ids)
      print(
          f"[AlpamayoServer] request frame_id={frame_id} "
          f"image_frame_ids={image_frame_ids} num_images={len(images)} "
          f"history={'real' if ego_history_xyz is not None and ego_history_rot is not None else 'stationary'} "
          f"mode={'vqa_native' if self.model.version == '1.5' else 'trajectory_prompt'} "
          f"prompt_chars={len(str(payload.get('prompt', '')))}",
          flush=True)
      result = self.model.generate(
          images=images,
          prompt=str(payload.get("prompt", "")),
          max_new_tokens=payload.get("max_new_tokens"),
          ego_history_xyz=ego_history_xyz,
          ego_history_rot=ego_history_rot,
          speed=payload.get("speed"),
          command=payload.get("command"),
          target_point=payload.get("target_point"),
          image_frame_ids=image_frame_ids)
      result["frame_id"] = frame_id
      result["image_frame_ids"] = image_frame_ids
      self._save_response_debug(request_dir, result)
      self._append_raw_response_log(frame_id, image_frame_ids, result)
      print(
          f"[AlpamayoServer] response frame_id={frame_id} "
          f"action={result.get('action')} reason={result.get('reason', '')[:120]} "
          f"raw={str(result.get('raw_response', ''))}",
          flush=True)
      self._write_json(result)
    except Exception as exc:  # pylint: disable=broad-except
      self._write_json(
          {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()},
          status=500)

  def log_message(self, fmt, *args):
    if _strtobool(os.environ.get("ALPAMAYO_SERVER_VERBOSE", "0")):
      super().log_message(fmt, *args)

  def _append_raw_response_log(self, frame_id, image_frame_ids, result: dict):
    if self.raw_log_path is None:
      return
    raw_response = str(result.get("raw_response", ""))
    reason = str(result.get("reason", ""))
    action = str(result.get("action", ""))
    record = (
        f"frame_id={frame_id} image_frame_ids={image_frame_ids} "
        f"action={action} reason={reason}\n"
        "raw_response:\n"
        f"{raw_response}\n"
        "----\n")
    with self.raw_log_lock:
      self.raw_log_path.parent.mkdir(parents=True, exist_ok=True)
      with self.raw_log_path.open("a", encoding="utf-8") as handle:
        handle.write(record)

  def _save_request_inputs(self, payload: dict, images: list[Image.Image], frame_id, image_frame_ids) -> Path | None:
    if self.input_save_dir is None:
      return None
    with self.input_save_lock:
      self.__class__.input_save_counter += 1
      request_index = self.__class__.input_save_counter
    safe_frame_id = "unknown" if frame_id is None else str(frame_id)
    request_dir = self.input_save_dir / f"{self.input_save_session}_{request_index:06d}_frame_{safe_frame_id}"
    request_dir.mkdir(parents=True, exist_ok=True)

    frame_ids = image_frame_ids if isinstance(image_frame_ids, list) else []
    for image_index, image in enumerate(images):
      image_frame_id = frame_ids[image_index] if image_index < len(frame_ids) else image_index
      image.save(request_dir / f"image_{image_index:02d}_frame_{image_frame_id}.png")

    metadata = {
        "request_index": request_index,
        "frame_id": frame_id,
        "image_frame_ids": image_frame_ids,
        "num_images": len(images),
        "speed": payload.get("speed"),
        "command": payload.get("command"),
        "target_point": payload.get("target_point"),
        "max_new_tokens": payload.get("max_new_tokens"),
        "prompt": payload.get("prompt", ""),
        "ego_history_xyz": payload.get("ego_history_xyz"),
        "ego_history_rot": payload.get("ego_history_rot"),
        "alpamayo_version": self.model.version if self.model is not None else None,
    }
    (request_dir / "request.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8")
    return request_dir

  @staticmethod
  def _save_response_debug(request_dir: Path | None, result: dict):
    if request_dir is None:
      return
    (request_dir / "response.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8")

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
  parser.add_argument("--alpamayo-version", default=os.environ.get("ALPAMAYO_VERSION", "auto"))
  parser.add_argument("--attn-implementation", default=os.environ.get("ALPAMAYO_ATTN_IMPLEMENTATION", ""))
  parser.add_argument("--vqa-top-p", type=float, default=float(os.environ.get("ALPAMAYO_VQA_TOP_P", 0.98)))
  parser.add_argument("--vqa-temperature", type=float, default=float(os.environ.get("ALPAMAYO_VQA_TEMPERATURE", 0.6)))
  parser.add_argument("--vqa-num-samples", type=int, default=int(os.environ.get("ALPAMAYO_VQA_NUM_SAMPLES", 1)))
  parser.add_argument(
      "--raw-log-path",
      default=os.environ.get(
          "ALPAMAYO_RAW_LOG_PATH",
          "results/alpamayo_vlm_test/local/alpamayo_raw_responses.txt"))
  parser.add_argument(
      "--raw-log-append",
      action="store_true",
      default=_strtobool(os.environ.get("ALPAMAYO_RAW_LOG_APPEND", "0")))
  parser.add_argument(
      "--input-save-dir",
      default=os.environ.get(
          "ALPAMAYO_INPUT_SAVE_DIR",
          "results/alpamayo_vlm_test/local/alpamayo_inputs"))
  parser.add_argument(
      "--raw-output-only",
      action="store_true",
      default=_strtobool(os.environ.get("ALPAMAYO_RAW_OUTPUT_ONLY", "0")))
  parser.add_argument("--do-sample", action="store_true", default=_strtobool(os.environ.get("ALPAMAYO_DO_SAMPLE", "0")))
  return parser.parse_args()


def main():
  args = parse_args()
  Handler.model = AlpamayoVLM(args)
  if args.raw_log_path:
    Handler.raw_log_path = Path(args.raw_log_path)
    Handler.raw_log_path.parent.mkdir(parents=True, exist_ok=True)
    if not args.raw_log_append:
      Handler.raw_log_path.write_text("", encoding="utf-8")
    print(f"Alpamayo raw responses logging to {Handler.raw_log_path}", flush=True)
  if args.input_save_dir:
    Handler.input_save_dir = Path(args.input_save_dir)
    Handler.input_save_dir.mkdir(parents=True, exist_ok=True)
    Handler.input_save_session = time.strftime("run_%Y%m%d_%H%M%S")
    Handler.input_save_counter = 0
    print(
        f"Alpamayo debug inputs logging to {Handler.input_save_dir} "
        f"with session {Handler.input_save_session}",
        flush=True)
  server = ThreadingHTTPServer((args.host, args.port), Handler)
  print(f"Alpamayo VLM server listening on http://{args.host}:{args.port}", flush=True)
  server.serve_forever()


if __name__ == "__main__":
  main()
