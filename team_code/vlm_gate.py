"""Asynchronous VLM gate for deciding whether steering intervention is allowed."""

from __future__ import annotations

import base64
from dataclasses import dataclass
import io
import json
import os
from pathlib import Path
import queue
import re
import sys
import threading
import time
import traceback
from typing import Optional
import urllib.error
import urllib.request

import numpy as np
from PIL import Image
import torch


@dataclass
class VLMDecision:
  frame_id: int
  enable_steering: bool
  steering_alpha: float = 0.0
  action: str = "none"
  reason: str = ""
  raw_response: str = ""
  timestamp: float = 0.0
  error: str = ""


@dataclass
class _VLMRequest:
  frame_id: int
  rgb_image: np.ndarray
  rgb_images: list[np.ndarray]
  image_frame_ids: list[int]
  speed: float
  command: int
  target_point: Optional[tuple[float, float]]
  ego_history_xyz: Optional[list]
  ego_history_rot: Optional[list]
  submitted_at: float


class AsyncVLMGate:
  """Runs VLM inference in a background thread and exposes the latest decision."""

  def __init__(self,
               model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
               every_n: int = 1,
               max_new_tokens: int = 256,
               device: str = "cuda:0",
               torch_dtype: str = "auto",
               quantization: str = "",
               backend: str = "auto",
               prompt: Optional[str] = None,
               verbose: bool = False,
               save_inputs: bool = False,
               input_save_dir: str = "vlm_inputs",
               binary_threshold: float = 0.5,
               confidence_threshold: float = 0.6,
               server_url: str = "http://127.0.0.1:8765/generate"):
    self.model_name = model_name
    self.every_n = max(int(every_n), 1)
    self.max_new_tokens = int(max_new_tokens)
    self.device = device
    self.torch_dtype = torch_dtype
    self.quantization = quantization.lower()
    self.backend = self._resolve_backend(backend, model_name)
    self.prompt = prompt or self._default_prompt()
    self.verbose = verbose
    self.save_inputs = save_inputs
    self.input_save_dir = Path(input_save_dir)
    self.binary_threshold = float(binary_threshold)
    self.confidence_threshold = float(confidence_threshold)
    self.server_url = server_url
    self.timing = str(os.environ.get("VLM_TIMING", "0")).lower() in ("1", "true", "yes", "y")
    if self.save_inputs:
      self.input_save_dir.mkdir(parents=True, exist_ok=True)

    self._requests: queue.Queue[_VLMRequest] = queue.Queue(maxsize=1)
    self._submit_lock = threading.Lock()
    self._frame_buffer: list[tuple[int, np.ndarray]] = []
    self._image_frame_offsets = self._parse_frame_offsets(
        os.environ.get("VLM_SERVER_FRAME_OFFSETS", "-6,-4,-2,0"))
    self._image_frame_buffer_size = max(
        max(abs(offset) for offset in self._image_frame_offsets) + 4,
        int(os.environ.get("VLM_SERVER_FRAME_BUFFER_SIZE", "12")))
    self._request_inflight = False
    self._last_request_frame_id: Optional[int] = None
    self._latest_lock = threading.Lock()
    self._latest: Optional[VLMDecision] = None
    self._stop_event = threading.Event()
    self._thread: Optional[threading.Thread] = None
    self._processor = None
    self._tokenizer = None
    self._model = None
    self._alpamayo_helper = None
    self._alpamayo_logits_processor_cls = None
    self._alpamayo_stop_after_eos_cls = None
    self._alpamayo_replace_padding_after_eos = None
    self._alpamayo_to_special_token = None
    self._internvl_pixel_dtype = torch.bfloat16
    self._previous_decision_image: Optional[Image.Image] = None
    self._previous_decision_frame_id: Optional[int] = None

  @classmethod
  def from_env(cls) -> "AsyncVLMGate":
    quantization = os.environ.get("VLM_QUANTIZATION", "").lower()
    if str(os.environ.get("VLM_LOAD_IN_4BIT", "0")).lower() in ("1", "true", "yes", "y"):
      quantization = "4bit"
    if str(os.environ.get("VLM_LOAD_IN_8BIT", "0")).lower() in ("1", "true", "yes", "y"):
      quantization = "8bit"

    return cls(
        model_name=os.environ.get("VLM_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct"),
        every_n=int(os.environ.get("VLM_EVERY_N", 1)),
        max_new_tokens=int(os.environ.get("VLM_MAX_NEW_TOKENS", 256)),
        device=os.environ.get("VLM_DEVICE", "cuda:0"),
        torch_dtype=os.environ.get("VLM_TORCH_DTYPE", "auto"),
        quantization=quantization,
        backend=os.environ.get("VLM_BACKEND", "auto"),
        prompt=os.environ.get("VLM_PROMPT", None),
        verbose=str(os.environ.get("VLM_VERBOSE", "1")).lower() in ("1", "true", "yes", "y"),
        save_inputs=str(os.environ.get("VLM_SAVE_INPUTS", "0")).lower() in ("1", "true", "yes", "y"),
        input_save_dir=os.environ.get("VLM_INPUT_SAVE_DIR", "vlm_inputs"),
        binary_threshold=float(os.environ.get("VLM_BINARY_THRESHOLD", 0.5)),
        confidence_threshold=float(os.environ.get("VLM_CONF_THRESHOLD", 0.6)),
        server_url=os.environ.get("VLM_SERVER_URL", os.environ.get("ALPAMAYO_SERVER_URL", "http://127.0.0.1:8765/generate")))

  def start(self):
    if self._thread is not None:
      return
    self._thread = threading.Thread(target=self._worker_loop, name="AsyncVLMGate", daemon=True)
    self._thread.start()

  def observe(self, frame_id, rgb_image):
    current_image = np.asarray(rgb_image).copy()
    with self._submit_lock:
      self._append_frame_to_buffer(int(frame_id), current_image)

  def submit(self, frame_id, rgb_image, speed, command, target_point=None,
             ego_history_xyz=None, ego_history_rot=None):
    submit_start = time.perf_counter()
    if isinstance(target_point, torch.Tensor):
      target_point = target_point.detach().flatten()[:2].cpu().numpy()
    if target_point is not None:
      target_point = tuple(float(x) for x in np.asarray(target_point).flatten()[:2])

    current_image = np.asarray(rgb_image).copy()
    with self._submit_lock:
      self._append_frame_to_buffer(int(frame_id), current_image)
      if not self._can_submit_request(int(frame_id)):
        return
      selected_frame_ids, selected_images = self._select_buffer_frames(
          int(frame_id),
          self._image_frame_offsets)
      self._request_inflight = True
      self._last_request_frame_id = int(frame_id)

    request = _VLMRequest(
        frame_id=int(frame_id),
        rgb_image=current_image,
        rgb_images=selected_images,
        image_frame_ids=selected_frame_ids,
        speed=float(speed),
        command=int(command),
        target_point=target_point,
        ego_history_xyz=ego_history_xyz,
        ego_history_rot=ego_history_rot,
        submitted_at=time.time())

    # if self.verbose:
    #   print(
    #       f"[VLMGate] submit frame={request.frame_id} speed={request.speed:.2f} "
    #       f"command={request.command} target_point={request.target_point}",
    #       flush=True)

    try:
      self._requests.put_nowait(request)
    except queue.Full:
      try:
        self._requests.get_nowait()
      except queue.Empty:
        pass
      try:
        self._requests.put_nowait(request)
      except queue.Full:
        self._mark_request_complete()
        return
    if self.timing:
      print(
          f"[VLMGateTiming] submit frame={request.frame_id} "
          f"elapsed_ms={(time.perf_counter() - submit_start) * 1000.0:.2f}",
          flush=True)

  def latest(self) -> Optional[VLMDecision]:
    with self._latest_lock:
      return self._latest

  def stop(self):
    self._stop_event.set()
    if self._thread is not None:
      self._thread.join(timeout=2.0)
      self._thread = None

  def _worker_loop(self):
    try:
      self._load_model()
    except Exception as exc:
      self._set_latest(VLMDecision(
          frame_id=-1,
          enable_steering=False,
          steering_alpha=0.0,
          action="none",
          reason="VLM initialization failed",
          timestamp=time.time(),
          error=f"{type(exc).__name__}: {exc}"))
      print(f"AsyncVLMGate failed to initialize:\n{traceback.format_exc()}", flush=True)
      return

    while not self._stop_event.is_set():
      try:
        request = self._requests.get(timeout=0.1)
      except queue.Empty:
        continue

      try:
        inference_start = time.perf_counter()
        decision = self._run_inference(request)
        inference_ms = (time.perf_counter() - inference_start) * 1000.0
      except Exception as exc:
        inference_ms = (time.perf_counter() - inference_start) * 1000.0 if "inference_start" in locals() else 0.0
        decision = VLMDecision(
            frame_id=request.frame_id,
            enable_steering=False,
            steering_alpha=0.0,
            action="none",
            reason="VLM inference failed",
            timestamp=time.time(),
            error=f"{type(exc).__name__}: {exc}")
        print(f"AsyncVLMGate inference failed:\n{traceback.format_exc()}", flush=True)

      self._set_latest(decision)
      if self.timing:
        print(
            f"[VLMGateTiming] result frame={decision.frame_id} "
            f"inference_ms={inference_ms:.2f} "
            f"age_ms={(time.time() - request.submitted_at) * 1000.0:.2f}",
            flush=True)
      if self.verbose:
        raw_response = decision.raw_response.replace("\n", " ")[:500]
        print(
            f"[VLMGate] result frame={decision.frame_id} "
            f"action={decision.action} reason={decision.reason} "
            f"error={decision.error} raw={raw_response}",
            flush=True)
      self._mark_request_complete()

  def _load_model(self):
    if self.backend == "alpamayo_server":
      print(f"AsyncVLMGate loaded backend=alpamayo_server url={self.server_url}", flush=True)
      return
    if self.backend == "alpamayo":
      self._load_alpamayo_model()
      return
    if self.backend == "internvl":
      self._load_internvl_model()
      return

    self._load_qwen_model()

  def _load_qwen_model(self):
    from transformers import AutoProcessor

    dtype = self._resolve_dtype(self.torch_dtype)
    self._processor = AutoProcessor.from_pretrained(self.model_name, trust_remote_code=True)
    model_cls = self._resolve_model_class()
    device_map = None if self.device == "cpu" else {"": self.device}
    model_kwargs = {
        "torch_dtype": dtype,
        "device_map": device_map,
        "trust_remote_code": True,
    }
    quantization_config = self._resolve_quantization_config(dtype)
    if quantization_config is not None:
      model_kwargs["quantization_config"] = quantization_config

    self._model = model_cls.from_pretrained(self.model_name, **model_kwargs)
    if self.device == "cpu":
      self._model.to("cpu")
    self._model.eval()
    quantization = self.quantization if self.quantization else "none"
    print(f"AsyncVLMGate loaded backend=qwen model={self.model_name} quantization={quantization}", flush=True)

  def _load_internvl_model(self):
    from transformers import AutoModel, AutoTokenizer

    dtype = self._resolve_dtype(self.torch_dtype)
    if dtype == "auto":
      dtype = torch.bfloat16
    self._internvl_pixel_dtype = dtype if self.device != "cpu" else torch.float32

    model_kwargs = {
        "torch_dtype": dtype,
        "low_cpu_mem_usage": True,
        "trust_remote_code": True,
    }
    use_flash_attn = str(os.environ.get("VLM_USE_FLASH_ATTN", "0")).lower() in ("1", "true", "yes", "y")
    if use_flash_attn:
      model_kwargs["use_flash_attn"] = True

    if self.quantization == "8bit":
      model_kwargs["load_in_8bit"] = True
    elif self.quantization == "4bit":
      model_kwargs["load_in_4bit"] = True
    if self.device != "cpu" and self.quantization in ("4bit", "8bit"):
      model_kwargs["device_map"] = {"": self.device}

    self._model = AutoModel.from_pretrained(self.model_name, **model_kwargs).eval()
    if self.device != "cpu" and self.quantization not in ("4bit", "8bit"):
      self._model.to(self.device)
    elif self.device == "cpu":
      self._model.to("cpu")

    self._tokenizer = AutoTokenizer.from_pretrained(
        self.model_name,
        trust_remote_code=True,
        use_fast=False)
    quantization = self.quantization if self.quantization else "none"
    print(f"AsyncVLMGate loaded backend=internvl model={self.model_name} quantization={quantization}", flush=True)

  def _load_alpamayo_model(self):
    self._ensure_alpamayo_on_path()

    from alpamayo_r1 import helper
    from alpamayo_r1.models.alpamayo_r1 import AlpamayoR1, ExpertLogitsProcessor
    from alpamayo_r1.models.token_utils import (
        StopAfterEOS,
        replace_padding_after_eos,
        to_special_token,
    )

    dtype = self._resolve_dtype(self.torch_dtype)
    if dtype == "auto":
      dtype = torch.bfloat16
    if self.quantization not in ("", "none", "0"):
      print("[VLMGate] Alpamayo backend ignores VLM_QUANTIZATION; load it on a suitable device.", flush=True)

    model_name = os.environ.get("VLM_ALPAMAYO_MODEL", self.model_name)
    self._model = AlpamayoR1.from_pretrained(model_name, dtype=dtype)
    if self.device != "cpu":
      self._model.to(self.device)
    else:
      self._model.to("cpu")
    self._model.eval()
    self._processor = helper.get_processor(self._model.tokenizer)
    self._tokenizer = self._model.tokenizer
    self._alpamayo_helper = helper
    self._alpamayo_logits_processor_cls = ExpertLogitsProcessor
    self._alpamayo_stop_after_eos_cls = StopAfterEOS
    self._alpamayo_replace_padding_after_eos = replace_padding_after_eos
    self._alpamayo_to_special_token = to_special_token
    print(f"AsyncVLMGate loaded backend=alpamayo model={model_name}", flush=True)

  def _run_inference(self, request: _VLMRequest) -> VLMDecision:
    image = self._to_pil_image(request.rgb_image)
    images = [self._to_pil_image(rgb_image) for rgb_image in request.rgb_images]
    if not images:
      images = [image]
    if str(os.environ.get("VLM_SERVER_TWO_FRAMES", "1")).lower() in ("0", "false", "no", "n"):
      images = [image]
    prompt = self._format_prompt(request)
    self._save_input_image(image, request, prompt)
    if self.backend == "alpamayo_server":
      return self._run_alpamayo_server_inference(images, prompt, request)
    if self.backend == "alpamayo":
      return self._run_alpamayo_inference(image, prompt, request.frame_id)
    if self.backend == "internvl":
      images = [image]
      if self._previous_decision_image is not None:
        images = [self._previous_decision_image, image]
      decision = self._run_internvl_inference(images, prompt, request.frame_id)
      self._previous_decision_image = image.copy()
      self._previous_decision_frame_id = request.frame_id
      return decision

    messages = [{
        "role": "user",
        "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": prompt},
        ],
    }]

    text = self._processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = self._processor(text=[text], images=[image], return_tensors="pt")
    inputs = {key: value.to(self._model.device) if hasattr(value, "to") else value for key, value in inputs.items()}

    with torch.inference_mode():
      generated_ids = self._model.generate(**inputs, max_new_tokens=self.max_new_tokens, do_sample=False)

    input_len = inputs["input_ids"].shape[1]
    generated_ids = generated_ids[:, input_len:]
    raw_response = self._processor.batch_decode(
        generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    decision = self._parse_response(raw_response, request.frame_id)
    decision.raw_response = raw_response
    decision.timestamp = time.time()
    return decision

  def _run_alpamayo_server_inference(self, images: list[Image.Image], prompt: str, request: _VLMRequest) -> VLMDecision:
    payload = {
        "frame_id": request.frame_id,
        "image_base64": self._image_to_base64_png(images[-1]),
        "images_base64": [self._image_to_base64_png(image) for image in images],
        "image_frame_ids": request.image_frame_ids if len(images) == len(request.image_frame_ids) else [request.frame_id],
        "prompt": prompt,
        "speed": request.speed,
        "command": request.command,
        "target_point": request.target_point,
        "ego_history_xyz": request.ego_history_xyz,
        "ego_history_rot": request.ego_history_rot,
        "max_new_tokens": self.max_new_tokens,
    }
    body = json.dumps(payload).encode("utf-8")
    http_request = urllib.request.Request(
        self.server_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST")
    timeout_s = float(os.environ.get("VLM_SERVER_TIMEOUT_S", 300.0))
    try:
      with urllib.request.urlopen(http_request, timeout=timeout_s) as response:
        response_data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
      error_body = exc.read().decode("utf-8", errors="replace")
      raise RuntimeError(f"Alpamayo server HTTP {exc.code}: {error_body}") from exc
    except urllib.error.URLError as exc:
      raise RuntimeError(f"Alpamayo server request failed: {exc}") from exc

    raw_response = str(response_data.get("raw_response", ""))
    if "action" in response_data:
      parse_payload = {
          "action": response_data.get("action"),
          "confidence": response_data.get("confidence", 1.0),
          "reason": response_data.get("reason", ""),
      }
      decision = self._parse_response(json.dumps(parse_payload), request.frame_id)
    else:
      decision = self._parse_response(raw_response, request.frame_id)
    decision.raw_response = raw_response or json.dumps(response_data)
    decision.timestamp = time.time()
    return decision

  @staticmethod
  def _parse_frame_offsets(spec: str) -> list[int]:
    offsets = []
    for item in str(spec).split(","):
      item = item.strip()
      if not item:
        continue
      offsets.append(int(item))
    if not offsets:
      return [-6, -4, -2, 0]
    return sorted(set(offsets))

  def _append_frame_to_buffer(self, frame_id: int, image: np.ndarray):
    if self._frame_buffer and self._frame_buffer[-1][0] == frame_id:
      self._frame_buffer[-1] = (frame_id, image.copy())
    else:
      self._frame_buffer.append((frame_id, image.copy()))
    if len(self._frame_buffer) > self._image_frame_buffer_size:
      self._frame_buffer = self._frame_buffer[-self._image_frame_buffer_size:]

  def _can_submit_request(self, frame_id: int) -> bool:
    if self._request_inflight:
      return False
    if self._last_request_frame_id is None:
      return True
    return frame_id - self._last_request_frame_id >= self.every_n

  def _mark_request_complete(self):
    with self._submit_lock:
      self._request_inflight = False

  def _select_buffer_frames(self, frame_id: int, offsets: list[int]) -> tuple[list[int], list[np.ndarray]]:
    selected_frame_ids = []
    selected_images = []
    for offset in offsets:
      target_frame = frame_id + int(offset)
      selected_frame_id, selected_image = self._nearest_buffer_frame(target_frame)
      selected_frame_ids.append(selected_frame_id)
      selected_images.append(selected_image.copy())
    return selected_frame_ids, selected_images

  def _nearest_buffer_frame(self, target_frame: int) -> tuple[int, np.ndarray]:
    if not self._frame_buffer:
      raise RuntimeError("VLM frame buffer is empty")
    past_or_current = [item for item in self._frame_buffer if item[0] <= target_frame]
    if past_or_current:
      return max(past_or_current, key=lambda item: item[0])
    return min(self._frame_buffer, key=lambda item: item[0])

  @staticmethod
  def _image_to_base64_png(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")

  def _run_alpamayo_inference(self, image: Image.Image, prompt: str, frame_id: int) -> VLMDecision:
    from transformers import StoppingCriteriaList
    from transformers.generation.logits_process import LogitsProcessorList

    messages = self._create_alpamayo_messages(image, prompt)
    inputs = self._processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
        continue_final_message=True,
        return_dict=True,
        return_tensors="pt",
    )
    ego_history_xyz, ego_history_rot = self._alpamayo_stationary_history()
    inputs = self._alpamayo_helper.to_device(inputs, self.device)
    input_ids = inputs.pop("input_ids")
    input_ids = self._model.fuse_traj_tokens(
        input_ids,
        {
            "ego_history_xyz": ego_history_xyz.to(self.device),
            "ego_history_rot": ego_history_rot.to(self.device),
        })

    eos_token_id = self._tokenizer.convert_tokens_to_ids(
        self._alpamayo_to_special_token("traj_future_start"))
    generation_config = self._model.vlm.generation_config
    generation_config.do_sample = str(os.environ.get("VLM_ALPAMAYO_DO_SAMPLE", "0")).lower() in ("1", "true", "yes", "y")
    generation_config.max_new_tokens = self.max_new_tokens
    generation_config.num_return_sequences = 1
    generation_config.output_logits = False
    generation_config.return_dict_in_generate = True
    generation_config.pad_token_id = self._tokenizer.pad_token_id
    logits_processor = LogitsProcessorList([
        self._alpamayo_logits_processor_cls(
            traj_token_offset=self._model.config.traj_token_start_idx,
            traj_vocab_size=self._model.config.traj_vocab_size,
        )
    ])

    with torch.inference_mode():
      outputs = self._model.vlm.generate(
          input_ids=input_ids,
          generation_config=generation_config,
          stopping_criteria=StoppingCriteriaList([
              self._alpamayo_stop_after_eos_cls(eos_token_id=eos_token_id)
          ]),
          logits_processor=logits_processor,
          **inputs)

    sequences = self._alpamayo_replace_padding_after_eos(
        token_ids=outputs.sequences.clone(),
        eos_token_id=eos_token_id,
        pad_token_id=self._tokenizer.pad_token_id)
    generated_ids = sequences[:, input_ids.shape[1]:]
    raw_response = self._tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False)[0]
    decision = self._parse_response(raw_response, frame_id)
    decision.raw_response = raw_response
    decision.timestamp = time.time()
    return decision

  def _ensure_alpamayo_on_path(self):
    candidates = []
    repo_path = os.environ.get("VLM_ALPAMAYO_REPO", os.environ.get("ALPAMAYO_REPO"))
    src_path = os.environ.get("VLM_ALPAMAYO_SRC", os.environ.get("ALPAMAYO_SRC"))
    if src_path:
      candidates.append(Path(src_path))
    if repo_path:
      candidates.append(Path(repo_path) / "src")
    candidates.append(Path(__file__).resolve().parents[2].parent / "alpamayo" / "src")

    for candidate in candidates:
      if candidate.exists():
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
          sys.path.insert(0, candidate_str)
        return

  def _create_alpamayo_messages(self, image: Image.Image, prompt: str):
    num_traj_token = int(os.environ.get("VLM_ALPAMAYO_HISTORY_PLACEHOLDER_TOKENS", 48))
    hist_traj_placeholder = (
        f"<|traj_history_start|>{'<|traj_history|>' * num_traj_token}<|traj_history_end|>"
    )
    text = (
        f"{hist_traj_placeholder}{prompt}\n"
        "Return final answer as strict JSON only: "
        "{\"action\":\"none|brake|left|right\",\"reason\":\"...\"}."
    )
    return [
        {
            "role": "system",
            "content": [{
                "type": "text",
                "text": "You are a driving assistant that generates safe and accurate actions.",
            }],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": text},
            ],
        },
        {
            "role": "assistant",
            "content": [{"type": "text", "text": "<|cot_start|>"}],
        },
    ]

  def _alpamayo_stationary_history(self):
    num_history_steps = int(os.environ.get("VLM_ALPAMAYO_HISTORY_STEPS", 16))
    ego_history_xyz = torch.zeros((1, 1, num_history_steps, 3), dtype=torch.float32)
    eye = torch.eye(3, dtype=torch.float32)
    ego_history_rot = eye.reshape(1, 1, 1, 3, 3).repeat(1, 1, num_history_steps, 1, 1)
    return ego_history_xyz, ego_history_rot

  def _run_internvl_inference(self, images: list[Image.Image], prompt: str, frame_id: int) -> VLMDecision:
    pixel_values_list = []
    num_patches_list = []
    for image in images:
      pixel_values = self._internvl_image_to_pixel_values(
          image,
          max_num=int(os.environ.get("VLM_INTERNVL_MAX_TILES_PER_IMAGE", "1")))
      pixel_values_list.append(pixel_values)
      num_patches_list.append(pixel_values.shape[0])
    pixel_values = torch.cat(pixel_values_list, dim=0)
    device = "cpu" if self.device == "cpu" else self.device
    pixel_values = pixel_values.to(device=device, dtype=self._internvl_pixel_dtype)
    if len(images) == 1:
      image_prefix = "Current frame: <image>\n"
    else:
      image_prefix = "Previous decision frame (t-1): <image>\nCurrent frame (t): <image>\n"
    question = image_prefix + prompt
    generation_config = {
        "max_new_tokens": self.max_new_tokens,
        "do_sample": False,
    }

    with torch.inference_mode():
      raw_response = self._model.chat(
          self._tokenizer,
          pixel_values,
          question,
          generation_config,
          num_patches_list=num_patches_list)

    decision = self._parse_response(raw_response, frame_id)
    decision.raw_response = raw_response
    decision.timestamp = time.time()
    return decision

  def _save_input_image(self, image: Image.Image, request: _VLMRequest, prompt: str):
    if not self.save_inputs:
      return

    timestamp_ms = int(time.time() * 1000)
    path = self.input_save_dir / f"frame_{request.frame_id:06d}_{timestamp_ms}.png"
    image.save(path)
    path.with_suffix(".prompt.txt").write_text(prompt, encoding="utf-8")
    if self.verbose:
      print(f"[VLMGate] saved input frame={request.frame_id} path={path}", flush=True)

  def _format_prompt(self, request: _VLMRequest) -> str:
    target_point = "unknown"
    if request.target_point is not None:
      target_point = f"({request.target_point[0]:.2f}, {request.target_point[1]:.2f})"
    nav_command = self._navigation_intent(request.command)
    temporal_context = self._temporal_context(request.frame_id)
    context = (
        f"Current speed is {request.speed:.2f} m/s, route command id is "
        f"{request.command} ({nav_command}), ego-frame target point is {target_point}. "
        f"{temporal_context}")
    formatted = self.prompt
    replacements = {
        "{CURRENT_SPEED}": f"{request.speed * 3.6:.1f}",
        "{CURRENT_SPEED_KM_H}": f"{request.speed * 3.6:.1f}",
        "{current_speed_km_h}": f"{request.speed * 3.6:.1f}",
        "{current_speed_m_s}": f"{request.speed:.2f}",
        "{speed_m_s}": f"{request.speed:.2f}",
        "{NAV_COMMAND}": nav_command,
        "{nav_command}": nav_command,
        "{navigation_intent}": nav_command,
        "{command}": str(request.command),
        "{target_point}": target_point,
        "{CURRENT_FRAME}": str(request.frame_id),
        "{current_frame}": str(request.frame_id),
        "{PREVIOUS_DECISION_FRAME}": (
            "none" if self._previous_decision_frame_id is None else str(self._previous_decision_frame_id)),
        "{previous_decision_frame}": (
            "none" if self._previous_decision_frame_id is None else str(self._previous_decision_frame_id)),
        "{FRAME_DELTA}": (
            "unknown" if self._previous_decision_frame_id is None
            else str(request.frame_id - self._previous_decision_frame_id)),
        "{frame_delta}": (
            "unknown" if self._previous_decision_frame_id is None
            else str(request.frame_id - self._previous_decision_frame_id)),
        "{TEMPORAL_CONTEXT}": temporal_context,
        "{temporal_context}": temporal_context,
    }
    has_context_placeholder = False
    for token, value in replacements.items():
      if token in formatted:
        has_context_placeholder = True
        formatted = formatted.replace(token, value)
    if has_context_placeholder:
      return formatted
    return f"{formatted}\n\n{context}"

  def _temporal_context(self, current_frame_id: int) -> str:
    if self._previous_decision_frame_id is None:
      return "Previous frame is unavailable."
    frame_delta = current_frame_id - self._previous_decision_frame_id
    return f"Previous frame is {frame_delta * 0.05:.2f} seconds ago."

  @staticmethod
  def _navigation_intent(command: int) -> str:
    return {
        1: "Turn Left at intersection",
        2: "Turn Right at intersection",
        3: "Go Straight",
        4: "Follow Lane",
        5: "Change Lane Left",
        6: "Change Lane Right",
    }.get(command, "Follow Lane")

  def _parse_response(self, response: str, frame_id: int) -> VLMDecision:
    data = self._extract_json(response)
    if data is None:
      action = self._text_to_action(response)
      reason = response.strip()[:160]
    else:
      action = self._extract_action(data)
      confidence = self._extract_confidence(data)
      if confidence is not None and confidence < self.confidence_threshold:
        action = "none"
      reason = self._format_reason(data)

    action = self._normalize_action(action)
    steering_alpha = 1.0 if action != "none" else 0.0
    return VLMDecision(
        frame_id=frame_id,
        enable_steering=steering_alpha > 0.0,
        steering_alpha=steering_alpha,
        action=action,
        reason=reason)

  @classmethod
  def _extract_command_alpha(cls, data: dict) -> float:
    command = data.get("cmd", data.get("high_level_command", ""))
    return 1.0 if cls._command_to_action(str(command).strip().lower()) != "none" else 0.0

  def _extract_action(self, data: dict) -> str:
    for key in (
        "action",
        "activation_action",
        "steering_action",
        "intervention_action",
        "cmd",
        "high_level_command",
        "command",
        "decision"):
      value = data.get(key)
      if value is None:
        continue
      action = self._normalize_action(str(value))
      if action != "none":
        return action
      if self._is_explicit_no_action(str(value)):
        return "none"

    action_vector = data.get("action_vector", data.get("activation_alpha", None))
    vector_action = self._action_from_vector(action_vector)
    if vector_action != "none":
      return vector_action

    score = self._extract_normalized_risk_score(data)
    if score >= self.binary_threshold:
      return self._fallback_action(data)
    return "none"

  @classmethod
  def _text_to_action(cls, text: str) -> str:
    lowered = text.lower()
    for pattern in (
        r"\baction\s*[:=]\s*['\"]?([a-zA-Z_ -]+)",
        r"\bcmd\s*[:=]\s*['\"]?([a-zA-Z_ -]+)",
        r"\bcommand\s*[:=]\s*['\"]?([a-zA-Z_ -]+)",
    ):
      match = re.search(pattern, lowered)
      if match:
        action = cls._normalize_action(match.group(1))
        if action != "none" or cls._is_explicit_no_action(match.group(1)):
          return action
    if re.search(r"\b(change|move|merge)\s+(to\s+)?(the\s+)?left\b", lowered):
      return "left"
    if re.search(r"\b(change|move|merge)\s+(to\s+)?(the\s+)?right\b", lowered):
      return "right"
    if re.search(r"\b(stop|brake|yield|emergency brake)\b", lowered):
      return "brake"
    return "none"

  @classmethod
  def _command_to_action(cls, command: str) -> str:
    normalized = re.sub(r"[^a-z_]+", "", command.lower())
    if normalized in ("brake_weak", "brakeweak", "weak_brake", "weakbrake", "gentle_brake", "gentlebrake"):
      return "brake_weak"
    if normalized in ("stop", "brake", "yield", "emergency_brake", "emergencybrake"):
      return "brake"
    if normalized in ("l_change", "lchange", "left", "left_change", "leftchange", "change_lane_left", "changelaneleft"):
      return "left"
    if normalized in ("r_change", "rchange", "right", "right_change", "rightchange", "change_lane_right", "changelaneright"):
      return "right"
    return "none"

  @classmethod
  def _normalize_action(cls, action: str | None) -> str:
    if action is None:
      return "none"
    normalized = cls._command_to_action(action)
    if normalized != "none":
      return normalized
    value = re.sub(r"[^a-z_]+", "", str(action).lower())
    if value in ("none", "no", "go", "keep", "normal", "no_steering", "nosteering", "no_action", "noaction"):
      return "none"
    return "none"

  @staticmethod
  def _is_explicit_no_action(value: str) -> bool:
    normalized = re.sub(r"[^a-z_]+", "", value.lower())
    return normalized in ("none", "no", "go", "keep", "normal", "no_steering", "nosteering", "no_action", "noaction")

  def _action_from_vector(self, value) -> str:
    if value is None:
      return "none"
    try:
      vector = np.asarray(value, dtype=np.float32).reshape(-1)
    except (TypeError, ValueError):
      return "none"
    if vector.size != 3:
      return "none"
    index = int(np.argmax(vector))
    if float(vector[index]) < self.binary_threshold:
      return "none"
    return ("brake", "left", "right")[index]

  def _fallback_action(self, data: dict) -> str:
    position = str(data.get("object_position", data.get("blocking_object_position", ""))).lower()
    if "left" in position:
      return "right"
    if "right" in position:
      return "left"
    return "brake"

  @classmethod
  def _extract_confidence(cls, data: dict) -> Optional[float]:
    if "confidence" not in data:
      return None
    value = cls._coerce_float(data.get("confidence"), 1.0)
    if value > 1.0:
      value = value / 100.0
    return float(np.clip(value, 0.0, 1.0))

  @classmethod
  def _extract_normalized_risk_score(cls, data: dict) -> float:
    zone_score = cls._extract_zone_score(data)
    if zone_score is not None:
      return zone_score

    if "risk_score" in data:
      return float(np.clip(cls._coerce_float(data.get("risk_score"), 0.0) / 100.0, 0.0, 1.0))
    if "risk" in data:
      return float(np.clip(cls._coerce_float(data.get("risk"), 0.0) / 100.0, 0.0, 1.0))

    value = cls._coerce_float(
        data.get(
            "steering_score",
            data.get("intervention_score", data.get("steering_alpha", data.get("alpha", 0.0)))),
        0.0)
    if value > 1.0:
      value = value / 100.0
    return float(np.clip(value, 0.0, 1.0))

  @classmethod
  def _extract_zone_score(cls, data: dict) -> Optional[float]:
    if cls._list_has_items(data.get("red_corridor")):
      return 1.0
    if cls._list_has_items(data.get("orange_corridor")):
      return 0.6
    if cls._list_has_items(data.get("yellow_corridor")):
      return 0.3
    if any(key in data for key in ("red_corridor", "orange_corridor", "yellow_corridor", "uncertain_objects")):
      return 0.0
    return None

  @staticmethod
  def _list_has_items(value) -> bool:
    if value is None:
      return False
    if isinstance(value, list):
      return len(value) > 0
    if isinstance(value, str):
      stripped = value.strip()
      return bool(stripped and stripped not in ("[]", "none", "None", "null"))
    return bool(value)

  @classmethod
  def _format_reason(cls, data: dict) -> str:
    parts = []
    zone_summary = cls._format_zone_reason(data)
    if zone_summary:
      parts.append(zone_summary)
    for key in (
        "hazard_scan",
        "corridor_intersection",
        "anomaly_hazard_check",
        "intersection_analysis",
        "intersect",
        "observation",
        "obs",
        "rsn",
        "cmd",
        "scene_semantics",
        "reasoning",
        "high_level_command",
        "reason"):
      value = data.get(key, "")
      if value is not None and str(value).strip():
        cleaned = re.sub(r"<\|[^|]+?\|>", "", str(value)).strip()
        if not cleaned:
          continue
        if key == "reason":
          parts.append(cleaned)
        else:
          parts.append(f"{key}: {cleaned}")
    return " | ".join(parts)[:240]

  @staticmethod
  def _format_zone_reason(data: dict) -> str:
    keys = ("red_corridor", "orange_corridor", "yellow_corridor", "uncertain_objects")
    if not any(key in data for key in keys):
      return ""
    red = data.get("red_corridor", [])
    orange = data.get("orange_corridor", [])
    yellow = data.get("yellow_corridor", [])
    uncertain = data.get("uncertain_objects", [])
    return (
        f"red_corridor={red} orange_corridor={orange} "
        f"yellow_corridor={yellow} uncertain={uncertain}")

  @staticmethod
  def _coerce_float(value, default: float) -> float:
    if value is None:
      return float(default)
    try:
      return float(value)
    except (TypeError, ValueError):
      match = re.search(r"-?\d+(?:\.\d+)?", str(value))
      if match is None:
        return float(default)
      try:
        return float(match.group(0))
      except ValueError:
        return float(default)

  def _set_latest(self, decision: VLMDecision):
    with self._latest_lock:
      self._latest = decision

  @staticmethod
  def _extract_json(response: str):
    try:
      return json.loads(response)
    except json.JSONDecodeError:
      pass

    match = re.search(r"\{.*\}", response, flags=re.DOTALL)
    if match is None:
      return None
    try:
      return json.loads(match.group(0))
    except json.JSONDecodeError:
      return None

  @staticmethod
  def _to_pil_image(rgb_image: np.ndarray) -> Image.Image:
    if rgb_image.dtype != np.uint8:
      rgb_image = np.clip(rgb_image, 0, 255).astype(np.uint8)
    if rgb_image.ndim == 3 and rgb_image.shape[0] == 3:
      rgb_image = np.transpose(rgb_image, (1, 2, 0))
    return Image.fromarray(rgb_image)

  @staticmethod
  def _resolve_model_class():
    try:
      from transformers import Qwen2_5_VLForConditionalGeneration
      return Qwen2_5_VLForConditionalGeneration
    except ImportError:
      pass
    try:
      from transformers import Qwen2VLForConditionalGeneration
      return Qwen2VLForConditionalGeneration
    except ImportError:
      pass
    from transformers import AutoModelForVision2Seq
    return AutoModelForVision2Seq

  @staticmethod
  def _resolve_backend(backend: str, model_name: str) -> str:
    backend = backend.lower()
    if backend == "auto":
      if "alpamayo" in model_name.lower():
        return "alpamayo"
      if "internvl" in model_name.lower():
        return "internvl"
      return "qwen"
    if backend not in ("qwen", "internvl", "alpamayo", "alpamayo_server"):
      raise ValueError("VLM_BACKEND must be one of: auto, qwen, internvl, alpamayo, alpamayo_server")
    return backend

  @staticmethod
  def _internvl_build_transform(input_size: int = 448):
    import torchvision.transforms as T
    from torchvision.transforms.functional import InterpolationMode

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB") if img.mode != "RGB" else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

  @staticmethod
  def _internvl_find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
      target_aspect_ratio = ratio[0] / ratio[1]
      ratio_diff = abs(aspect_ratio - target_aspect_ratio)
      if ratio_diff < best_ratio_diff:
        best_ratio_diff = ratio_diff
        best_ratio = ratio
      elif ratio_diff == best_ratio_diff:
        if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
          best_ratio = ratio
    return best_ratio

  @classmethod
  def _internvl_dynamic_preprocess(cls, image, min_num=1, max_num=12, image_size=448, use_thumbnail=True):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height
    target_ratios = set(
        (i, j)
        for n in range(min_num, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
    target_aspect_ratio = cls._internvl_find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
      box = (
          (i % (target_width // image_size)) * image_size,
          (i // (target_width // image_size)) * image_size,
          ((i % (target_width // image_size)) + 1) * image_size,
          ((i // (target_width // image_size)) + 1) * image_size,
      )
      processed_images.append(resized_img.crop(box))
    if use_thumbnail and len(processed_images) != 1:
      processed_images.append(image.resize((image_size, image_size)))
    return processed_images

  @classmethod
  def _internvl_image_to_pixel_values(cls, image: Image.Image, input_size: int = 448, max_num: int = 12):
    transform = cls._internvl_build_transform(input_size=input_size)
    images = cls._internvl_dynamic_preprocess(
        image,
        image_size=input_size,
        use_thumbnail=True,
        max_num=max_num)
    pixel_values = [transform(tile) for tile in images]
    return torch.stack(pixel_values)

  def _resolve_quantization_config(self, dtype):
    if self.quantization in ("", "none", "0"):
      return None
    if self.quantization not in ("4bit", "8bit"):
      raise ValueError("VLM_QUANTIZATION must be one of: 4bit, 8bit, none")
    if self.device == "cpu":
      raise ValueError("bitsandbytes quantized VLM inference requires a CUDA device")

    from transformers import BitsAndBytesConfig

    if self.quantization == "8bit":
      return BitsAndBytesConfig(load_in_8bit=True)

    compute_dtype = dtype
    if compute_dtype == "auto":
      compute_dtype = torch.bfloat16
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=compute_dtype,
        bnb_4bit_use_double_quant=True)

  @staticmethod
  def _resolve_dtype(dtype_name: str):
    if dtype_name == "auto":
      return "auto"
    if dtype_name in ("float16", "fp16"):
      return torch.float16
    if dtype_name in ("bfloat16", "bf16"):
      return torch.bfloat16
    if dtype_name in ("float32", "fp32"):
      return torch.float32
    return "auto"

  @staticmethod
  def _default_prompt() -> str:
    return """
"""
# Context:
# # - Current speed: {CURRENT_SPEED} km/h.
# # - Navigation intent: {NAV_COMMAND}.
# # - Ego-frame target point: {target_point}.
# # - Current frame: {CURRENT_FRAME}; previous decision frame: {PREVIOUS_DECISION_FRAME}; frame delta: {FRAME_DELTA}.
# # - {TEMPORAL_CONTEXT}

# Safety policy:
# - Identify hazards blocking the ego lane or intended path.
# - Consider whether obstacles are moving into, staying in, or clearing the ego path.
# - Use LEFT or RIGHT only if that side has clearly safer free space.
# - Prefer BRAKE over lateral steering when lane clearance is uncertain.
# - Prefer NONE when there is no immediate hazard.
