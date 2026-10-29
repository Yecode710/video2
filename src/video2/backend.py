"""Local Hugging Face Qwen2.5-VL inference with explicit video timing."""

from dataclasses import dataclass
import random

from .video import SampledVideo


@dataclass(frozen=True)
class ModelConfig:
    model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    revision: str = "main"
    device: str = "auto"
    dtype: str = "auto"
    max_new_tokens: int = 256
    seed: int = 42
    min_pixels: int = 128 * 28 * 28
    max_pixels: int = 256 * 28 * 28


class QwenBackend:
    def __init__(self, config: ModelConfig):
        try:
            import numpy as np
            import torch
            import transformers
            from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        except ImportError as error:
            raise RuntimeError("Install inference dependencies: pip install -e '.[inference]'") from error
        if config.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("--device cuda requested but CUDA is unavailable")
        if config.dtype == "float16" and (
            config.device == "cpu" or (config.device == "auto" and not torch.cuda.is_available())
        ):
            raise ValueError("Use float32 or auto on CPU; float16 CPU inference is unsupported here")
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
            torch.backends.cudnn.benchmark = False
        self.torch = torch
        self.config = config
        kwargs = {"revision": config.revision, "attn_implementation": "sdpa"}
        if config.dtype != "auto":
            kwargs["dtype"] = getattr(torch, config.dtype)
        else:
            kwargs["dtype"] = "auto"
        if config.device == "auto":
            kwargs["device_map"] = "auto"
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(config.model, **kwargs)
        if config.device != "auto":
            self.model.to(config.device)
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(
            config.model,
            revision=config.revision,
            use_fast=False,
            min_pixels=config.min_pixels,
            max_pixels=config.max_pixels,
        )
        # In transformers 4.57.1 from_pretrained overrides min/max attributes,
        # but the video resize implementation reads `size` directly.
        self.processor.video_processor.size = {
            "shortest_edge": config.min_pixels, "longest_edge": config.max_pixels,
        }
        self.metadata = {
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "model_commit": getattr(self.model.config, "_commit_hash", None),
            "model_dtype": str(self.model.dtype),
            "device_map": {key: str(value) for key, value in getattr(self.model, "hf_device_map", {}).items()},
        }

    def summarize(self, video: SampledVideo, prompt: str) -> str:
        # The placeholder establishes video placement; decoded RGB data is supplied
        # directly below. Never put references or other dataset fields in messages.
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": prompt}
        ]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(
            text=[text], videos=[video.frames], fps=[video.effective_fps], do_sample_frames=False,
            padding=True, return_tensors="pt",
        )
        inputs = inputs.to(self.model.device)
        with self.torch.inference_mode():
            tokens = self.model.generate(
                **inputs, max_new_tokens=self.config.max_new_tokens,
                do_sample=False, num_beams=1, use_cache=True,
            )
        generated = tokens[:, inputs["input_ids"].shape[1]:]
        summary = self.processor.batch_decode(
            generated, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0].strip()
        if not summary:
            raise RuntimeError("Model returned an empty summary")
        return summary
