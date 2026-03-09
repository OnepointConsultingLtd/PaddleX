# Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Transformers-based Vision-Language Model predictor.

Drop-in replacement for the PaddlePaddle ``DocVLMPredictor`` used by the
PaddleOCR-VL pipeline.  Works with any HuggingFace VLM that follows the
Qwen2-VL / Qwen2.5-VL interface (``AutoModelForVision2Seq`` +
``AutoProcessor``).  No PaddlePaddle installation is required.

Config example (inside the pipeline YAML under ``SubModules.VLRecognition``)::

    VLRecognition:
      use_hf_backend: true
      model_dir: /path/to/Qwen2.5-VL-3B   # local dir or HF repo id
      device: auto                          # "cpu", "gpu", "gpu:0", "auto"
      batch_size: 8
      torch_dtype: bfloat16                # float32 | float16 | bfloat16
"""

from __future__ import annotations

from typing import Dict, Iterator, List, Optional

import numpy as np


class _BatchSampler:
    """Minimal stub so the pipeline can read ``vl_rec_model.batch_sampler.batch_size``."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size


class HFVLMPredictor:
    """HuggingFace Vision-Language Model predictor for the PaddleOCR-VL pipeline.

    Wraps any ``AutoModelForVision2Seq``-compatible model (e.g. Qwen2-VL,
    Qwen2.5-VL, InternVL2, etc.) and exposes the same ``predict()`` API as the
    PaddlePaddle ``DocVLMPredictor``.

    The pipeline calls::

        results = list(
            self.vl_rec_model.predict(
                [{"image": bgr_np_array, "query": "OCR:"},  ...],
                skip_special_tokens=True,
                use_cache=True,
                min_pixels=112896,
                max_pixels=1003520,
                max_new_tokens=4096,
            )
        )

    Each yielded item is a ``dict`` with at least a ``"result"`` key that
    contains the generated text.

    Parameters
    ----------
    model_dir:
        HuggingFace repo id **or** local directory containing the model.
    device:
        Inference device.  ``"auto"`` lets ``accelerate`` place layers
        automatically.  Other accepted values: ``"cpu"``, ``"gpu"``,
        ``"gpu:0"``, ``"cuda"``, ``"cuda:0"``.
    batch_size:
        Exposed as ``batch_sampler.batch_size`` for queue-mode compatibility.
        Image batching inside ``predict()`` is controlled by the pipeline.
    torch_dtype:
        Weight dtype string: ``"float32"``, ``"float16"``, or ``"bfloat16"``.
        ``"bfloat16"`` is recommended for modern GPUs.
    min_pixels:
        Default minimum pixel budget for dynamic-resolution image processing
        (Qwen2-VL style).  Overridden per call if the caller supplies it.
    max_pixels:
        Default maximum pixel budget.  Overridden per call if the caller
        supplies it.
    """

    def __init__(
        self,
        model_dir: str,
        device: Optional[str] = "auto",
        batch_size: int = 1,
        torch_dtype: str = "bfloat16",
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFVLMPredictor requires 'torch'. Install it from https://pytorch.org/get-started/locally/"
            ) from exc

        try:
            from transformers import AutoProcessor

            # transformers>=5.0 renamed AutoModelForVision2Seq to
            # AutoModelForImageTextToText; fall back gracefully.
            try:
                from transformers import AutoModelForImageTextToText as _AutoVLM
            except ImportError:
                from transformers import AutoModelForVision2Seq as _AutoVLM  # type: ignore[assignment]
            AutoModelForVision2Seq = _AutoVLM
        except ImportError as exc:
            raise ImportError(
                f"HFVLMPredictor failed to import from 'transformers': {exc}\n\n"
                "Install transformers >= 4.39.0:\n"
                "    pip install 'transformers>=4.39.0'"
            ) from exc

        self._torch = torch
        self.batch_sampler = _BatchSampler(batch_size)
        self._default_min_pixels = min_pixels or 112_896
        self._default_max_pixels = max_pixels or 1_003_520

        _dtype_map = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }
        _torch_dtype = _dtype_map.get(torch_dtype, torch.bfloat16)
        _device_map = self._parse_device(device)

        self._processor = AutoProcessor.from_pretrained(
            model_dir, trust_remote_code=True
        )
        self._model = AutoModelForVision2Seq.from_pretrained(
            model_dir,
            torch_dtype=_torch_dtype,
            device_map=_device_map,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self._model.eval()

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def predict(
        self,
        inputs: List[Dict],
        skip_special_tokens: bool = True,
        use_cache: bool = True,
        min_pixels: Optional[int] = None,
        max_pixels: Optional[int] = None,
        max_new_tokens: int = 4096,
        repetition_penalty: Optional[float] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        **kwargs,
    ) -> Iterator[Dict]:
        """Run VLM inference on a list of ``{"image": ..., "query": ...}`` dicts.

        Parameters
        ----------
        inputs:
            Each element must have:
            - ``"image"``: BGR ``numpy.ndarray`` (OpenCV format).
            - ``"query"``: Prompt string, e.g. ``"OCR:"``.
        skip_special_tokens:
            Passed to the tokenizer ``decode()`` call.
        use_cache:
            Whether to use KV-cache during generation.
        min_pixels / max_pixels:
            Image resolution bounds for dynamic-resolution models (Qwen2-VL).
            Defaults to the values set in ``__init__``.
        max_new_tokens:
            Maximum number of tokens to generate.
        repetition_penalty, temperature, top_p:
            Sampling parameters forwarded to ``model.generate()``.

        Yields
        ------
        dict
            A result dict with a ``"result"`` key containing the decoded text.
        """
        import torch
        from PIL import Image as _PIL_Image

        min_px = min_pixels if min_pixels is not None else self._default_min_pixels
        max_px = max_pixels if max_pixels is not None else self._default_max_pixels

        gen_kwargs: Dict = {"use_cache": use_cache, "max_new_tokens": max_new_tokens}
        if repetition_penalty is not None:
            gen_kwargs["repetition_penalty"] = repetition_penalty
        if temperature is not None:
            gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p

        for item in inputs:
            img_bgr: np.ndarray = item["image"]
            query: str = item["query"]

            pil_img = _PIL_Image.fromarray(img_bgr[:, :, ::-1])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": query},
                    ],
                }
            ]

            try:
                text = self._processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )

                # Build processor kwargs — only pass pixel limits when the
                # processor's image processor actually supports them (Qwen2-VL).
                proc_kwargs: Dict = {}
                img_proc = getattr(self._processor, "image_processor", None)
                if img_proc is not None and hasattr(img_proc, "min_pixels"):
                    proc_kwargs["min_pixels"] = min_px
                    proc_kwargs["max_pixels"] = max_px

                model_inputs = self._processor(
                    text=[text],
                    images=[pil_img],
                    return_tensors="pt",
                    **proc_kwargs,
                )
                # Move all tensors to the model's device
                _device = next(self._model.parameters()).device
                model_inputs = {
                    k: v.to(_device) if hasattr(v, "to") else v
                    for k, v in model_inputs.items()
                }

                with torch.no_grad():
                    generated_ids = self._model.generate(
                        **model_inputs, **gen_kwargs
                    )

                input_token_len = model_inputs["input_ids"].shape[1]
                new_token_ids = generated_ids[:, input_token_len:]
                result_text = self._processor.decode(
                    new_token_ids[0], skip_special_tokens=skip_special_tokens
                )
                yield {"result": result_text.strip()}

            except Exception:
                yield {"result": ""}

    def close(self) -> None:
        """Release GPU memory held by the model."""
        if hasattr(self, "_model"):
            del self._model
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_device(device: Optional[str]) -> str:
        """Normalise PaddlePaddle / CUDA device strings to ``device_map`` format."""
        if not device or device == "auto":
            return "auto"
        if device == "cpu":
            return "cpu"
        if device.startswith("gpu"):
            parts = device.split(":", 1)
            return f"cuda:{parts[1]}" if len(parts) > 1 else "cuda"
        return device  # pass-through: "cuda", "cuda:0", etc.
