#!/usr/bin/env python3
"""Minimal Qwen-VL inference script.

Running this once will download the selected HuggingFace model into the local
HF cache. Pass any image path to do a small JSON steering-alpha test.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from PIL import Image
import torch
from transformers import AutoProcessor

TEAM_CODE_ROOT = Path(__file__).resolve().parents[1] / "team_code"
sys.path.insert(0, str(TEAM_CODE_ROOT))
from vlm_gate import AsyncVLMGate  # noqa: E402


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
  parser.add_argument("--image", required=True, help="Path to a front-camera image.")
  parser.add_argument("--backend", choices=["auto", "qwen", "internvl"], default="auto")
  parser.add_argument("--device", default="cuda:0")
  parser.add_argument("--quantization", choices=["none", "4bit", "8bit"], default="none")
  parser.add_argument("--max-new-tokens", type=int, default=256)
  parser.add_argument("--prompt-file", default=None, help="Use the exact prompt text saved by vlm_gate.")
  parser.add_argument(
      "--prompt",
      default="""

You are an autonomous driving safety evaluator.
The image shows a 3D safety corridor outlining the ego vehicle's exact future path over 4.0 seconds:
- Deep red corridor: 0.0s to 1.5s (Critical)
- Orange corridor: 1.5s to 3.0s (Warning)
- Yellow corridor: 3.0s to 4.0s (Anticipation)

CRITICAL INSTRUCTION:
Look extremely closely inside and at the edges of the colored corridors. 

Output strictly as JSON. Follow this step-by-step logic:
{
  "hazard_scan": "<Scan the entire image. List any pedestrian, vehicle, or obstacle found inside the corridors. Be specific.>",
  "corridor_intersection": "<State exactly which colored corridor (Red, Orange, Yellow, or None) the identified hazard is touching or standing inside.>",
  "risk": <Integer ONLY. MUST be 100 if intersection is Red. MUST be 70 if intersection is Orange. MUST be 50 if intersection is Yellow. MUST be 0 if None.>
}
""",
  )
  return parser.parse_args()


def resolve_model_class():
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


def resolve_quantization_config(quantization, device):
  if quantization == "none":
    return None
  if device == "cpu":
    raise ValueError("bitsandbytes quantized inference requires a CUDA device")

  from transformers import BitsAndBytesConfig

  if quantization == "8bit":
    return BitsAndBytesConfig(load_in_8bit=True)

  return BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16,
      bnb_4bit_use_double_quant=True)


def main():
  args = parse_args()
  image_path = Path(args.image)
  image = Image.open(image_path).convert("RGB")
  prompt = args.prompt
  if args.prompt_file is not None:
    prompt = Path(args.prompt_file).read_text(encoding="utf-8")
  backend = AsyncVLMGate._resolve_backend(args.backend, args.model)

  if backend == "internvl":
    run_internvl(args, image, prompt)
    return

  processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
  model_cls = resolve_model_class()
  device_map = None if args.device == "cpu" else {"": args.device}
  model_kwargs = {
      "torch_dtype": "auto",
      "device_map": device_map,
      "trust_remote_code": True,
  }
  quantization_config = resolve_quantization_config(args.quantization, args.device)
  if quantization_config is not None:
    model_kwargs["quantization_config"] = quantization_config
  model = model_cls.from_pretrained(args.model, **model_kwargs)
  if args.device == "cpu":
    model.to("cpu")
  model.eval()

  messages = [{
      "role": "user",
      "content": [
          {"type": "image", "image": image},
          {"type": "text", "text": prompt},
      ],
  }]

  text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
  inputs = processor(text=[text], images=[image], return_tensors="pt")
  inputs = {
      key: value.to(model.device) if hasattr(value, "to") else value
      for key, value in inputs.items()
  }

  with torch.inference_mode():
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=False)

  input_len = inputs["input_ids"].shape[1]
  generated_ids = generated_ids[:, input_len:]
  response = processor.batch_decode(
      generated_ids,
      skip_special_tokens=True,
      clean_up_tokenization_spaces=False)[0]
  print(response)


def run_internvl(args, image, prompt):
  from transformers import AutoModel, AutoTokenizer

  model_kwargs = {
      "torch_dtype": torch.bfloat16,
      "low_cpu_mem_usage": True,
      "trust_remote_code": True,
  }
  if args.quantization == "8bit":
    model_kwargs["load_in_8bit"] = True
  elif args.quantization == "4bit":
    model_kwargs["load_in_4bit"] = True
  if args.device != "cpu" and args.quantization in ("4bit", "8bit"):
    model_kwargs["device_map"] = {"": args.device}

  model = AutoModel.from_pretrained(args.model, **model_kwargs).eval()
  if args.device != "cpu" and args.quantization == "none":
    model.to(args.device)
  elif args.device == "cpu":
    model.to("cpu")
  tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True, use_fast=False)
  pixel_values = AsyncVLMGate._internvl_image_to_pixel_values(image)
  dtype = torch.float32 if args.device == "cpu" else torch.bfloat16
  device = "cpu" if args.device == "cpu" else args.device
  pixel_values = pixel_values.to(device=device, dtype=dtype)
  question = f"<image>\n{prompt}"
  generation_config = {
      "max_new_tokens": args.max_new_tokens,
      "do_sample": False,
  }
  with torch.inference_mode():
    response = model.chat(tokenizer, pixel_values, question, generation_config)
  print(response)


if __name__ == "__main__":
  main()
