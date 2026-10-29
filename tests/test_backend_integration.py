"""Offline real-Transformers CPU interface test, using a tiny random model.

This exercises video preprocessing and generation, never summary quality or the
pretrained 7B checkpoint. No weights, tokenizer files, or other assets are fetched.
"""

import importlib.util
import unittest
from unittest.mock import patch

from video2.backend import ModelConfig, QwenBackend
from video2.video import SampledVideo


INFERENCE_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None
    for name in ("torch", "torchvision", "transformers", "tokenizers", "numpy")
)


@unittest.skipUnless(INFERENCE_DEPS_AVAILABLE, "optional inference dependencies not installed")
class RealBackendIntegrationTests(unittest.TestCase):
    def test_real_processor_and_random_model_generate_on_cpu(self):
        import numpy as np
        import torch
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import (
            Qwen2TokenizerFast,
            Qwen2VLImageProcessor,
            Qwen2VLVideoProcessor,
            Qwen2_5_VLConfig,
            Qwen2_5_VLForConditionalGeneration,
            Qwen2_5_VLProcessor,
        )

        vocab = {
            token: index
            for index, token in enumerate(
                [
                    "<|endoftext|>", "<|vision_start|>", "<|vision_end|>",
                    "<|video_pad|>", "<|image_pad|>", "<|im_start|>", "<|im_end|>",
                    "user", "assistant", "Summarize", "video", "visible", "action", "summary",
                ]
            )
        }
        engine = Tokenizer(models.WordLevel(vocab, unk_token="<|endoftext|>"))
        engine.pre_tokenizer = pre_tokenizers.WhitespaceSplit()
        tokenizer = Qwen2TokenizerFast(
            tokenizer_object=engine, additional_special_tokens=list(vocab)[:7]
        )
        template = (
            "{% for message in messages %}{% for item in message['content'] %}"
            "{% if item['type'] == 'video' %}"
            "<|vision_start|><|video_pad|><|vision_end|>"
            "{% elif item['type'] == 'text' %} {{ item['text'] }}"
            "{% endif %}{% endfor %}{% endfor %}"
        )
        processor = Qwen2_5_VLProcessor(
            image_processor=Qwen2VLImageProcessor(min_pixels=56 * 56, max_pixels=56 * 56),
            video_processor=Qwen2VLVideoProcessor(min_pixels=56 * 56, max_pixels=56 * 56),
            tokenizer=tokenizer,
            chat_template=template,
        )
        video = SampledVideo(
            frames=[np.full((56, 56, 3), value, dtype=np.uint8) for value in (0, 50, 100, 150)],
            timestamps=[0.0, 2.0, 4.0, 6.0],
            duration_seconds=8.0,
            source_fps=30.0,
            effective_fps=0.5,
        )
        messages = [{"role": "user", "content": [
            {"type": "video"}, {"type": "text", "text": "Summarize video"}
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text], videos=[video.frames], fps=[video.effective_fps],
            do_sample_frames=False, padding=True, return_tensors="pt",
        )
        self.assertEqual(inputs["video_grid_thw"].tolist(), [[2, 4, 4]])
        self.assertEqual(tuple(inputs["pixel_values_videos"].shape), (32, 1176))
        self.assertEqual(inputs["second_per_grid_ts"].tolist(), [4.0])
        self.assertEqual((inputs["input_ids"] == vocab["<|video_pad|>"]).sum().item(), 8)

        config = Qwen2_5_VLConfig(
            text_config={
                "vocab_size": len(tokenizer), "hidden_size": 32, "intermediate_size": 64,
                "num_hidden_layers": 1, "num_attention_heads": 4, "num_key_value_heads": 2,
                "max_position_embeddings": 256,
                "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]},
                "bos_token_id": 0, "eos_token_id": 0, "pad_token_id": 0,
            },
            vision_config={
                "depth": 1, "hidden_size": 32, "intermediate_size": 64,
                "num_heads": 4, "out_hidden_size": 32, "patch_size": 14,
                "temporal_patch_size": 2, "spatial_merge_size": 2,
                "window_size": 112, "fullatt_block_indexes": [0],
            },
            image_token_id=vocab["<|image_pad|>"],
            video_token_id=vocab["<|video_pad|>"],
            vision_start_token_id=vocab["<|vision_start|>"],
            vision_end_token_id=vocab["<|vision_end|>"],
        )
        previous_threads = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            with torch.random.fork_rng(devices=[]):
                torch.manual_seed(42)
                model = Qwen2_5_VLForConditionalGeneration(config).eval()
            # Random weights may emit only special tokens. Restrict to ordinary
            # tiny-vocabulary tokens so the production empty-summary guard remains
            # active while this test checks tensor shapes and generation plumbing.
            model.generation_config.suppress_tokens = list(range(7))
            # Exercise production loading/overrides without fetching a checkpoint.
            # Simulate an old cached processor with a conflicting pixel budget.
            processor.video_processor.size = {"shortest_edge": 56 * 56, "longest_edge": 448 * 448}
            with patch("transformers.Qwen2_5_VLForConditionalGeneration.from_pretrained", return_value=model), \
                    patch("transformers.AutoProcessor.from_pretrained", return_value=processor):
                backend = QwenBackend(ModelConfig(
                    device="cpu", dtype="float32", max_new_tokens=2,
                    min_pixels=56 * 56, max_pixels=56 * 56,
                ))
            self.assertEqual(backend.processor.video_processor.size,
                             {"shortest_edge": 56 * 56, "longest_edge": 56 * 56})
            with patch.object(model, "generate", wraps=model.generate) as generate:
                summary = backend.summarize(video, "Summarize video")
            self.assertTrue(summary.strip())
            self.assertEqual(len(summary.split()), 2)
            generate.assert_called_once()
            call = generate.call_args.kwargs
            self.assertEqual(call["second_per_grid_ts"].tolist(), [4.0])
            self.assertEqual(call["max_new_tokens"], 2)
            self.assertFalse(call["do_sample"])
            self.assertEqual(call["num_beams"], 1)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
