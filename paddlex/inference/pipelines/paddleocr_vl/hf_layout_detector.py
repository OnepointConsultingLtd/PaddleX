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

"""Transformers-based PP-DocLayout layout detector (V2 and V3).

Drop-in replacement for the PaddlePaddle LayoutAnalysisPredictor used inside
the PaddleOCR-VL pipeline.  Requires only ``torch`` and ``transformers``; no
PaddlePaddle installation is needed.

Config example (inside the pipeline YAML under ``SubModules.LayoutDetection``)::

    LayoutDetection:
      use_hf_backend: true
      model_version: v3          # "v2" or "v3"  (default: v3)
      model_dir: PaddlePaddle/PP-DocLayoutV3_safetensors
      device: cpu
      batch_size: 1
      threshold: 0.5
      layout_nms: true

V2 vs V3 differences
--------------------
* **V2** (``PPDocLayoutV2*``): outputs bounding boxes only.  No masks or
  polygon points.  Use model ``PaddlePaddle/PP-DocLayoutV2_safetensors``.
* **V3** (``PPDocLayoutV3*``): outputs bounding boxes **and** polygon points
  derived from per-instance segmentation masks.  Use model
  ``PaddlePaddle/PP-DocLayoutV3_safetensors``.

The ``model_version`` key selects which transformers classes are loaded.
When omitted the detector auto-detects the version from the model's
``architectures`` field in ``config.json``.
"""

from __future__ import annotations

import threading
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np

# cuDNN batch-norm (and some other ops) is not thread-safe when multiple
# threads call into the same GPU simultaneously.  This module-level lock
# serialises the CUDA forward pass so pool-size > 2 works reliably.
_GPU_LOCK = threading.Lock()


class _BatchSampler:
    """Minimal batch-sampler stub for pipeline queue-depth compatibility."""

    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size


# ---------------------------------------------------------------------------
# Model-version registry
# ---------------------------------------------------------------------------

_V2_MODEL_CLS = "PPDocLayoutV2ForObjectDetection"
_V3_MODEL_CLS = "PPDocLayoutV3ForObjectDetection"

_VERSION_MAP: Dict[str, Dict[str, str]] = {
    "v2": {
        "model_cls": "PPDocLayoutV2ForObjectDetection",
        "processor_cls": "PPDocLayoutV2ImageProcessorFast",
    },
    "v3": {
        "model_cls": "PPDocLayoutV3ForObjectDetection",
        "processor_cls": "PPDocLayoutV3ImageProcessorFast",
    },
}


def _detect_version_from_config(model_dir: str) -> str:
    """Peek at ``config.json`` to decide whether this is a V2 or V3 model."""
    try:
        from transformers import AutoConfig

        cfg = AutoConfig.from_pretrained(model_dir)
        archs = getattr(cfg, "architectures", []) or []
        for arch in archs:
            if _V2_MODEL_CLS in arch:
                return "v2"
            if _V3_MODEL_CLS in arch:
                return "v3"
    except Exception:
        pass
    # Default to V3 (newer, more capable model)
    return "v3"


def _load_model_and_processor(model_dir: str, version: str, device: str = "cpu"):
    """Import and instantiate the correct transformers classes for *version*.

    *device* is forwarded to ``from_pretrained`` so weights land on the target
    device immediately, avoiding the meta-tensor error that occurs when
    ``model.to(device)`` is called after ``accelerate`` has already placed
    tensors on the "meta" device during a concurrent load.
    """
    entry = _VERSION_MAP.get(version)
    if entry is None:
        raise ValueError(
            f"Unknown model_version {version!r}. Choose 'v2' or 'v3'."
        )

    import importlib

    transformers = importlib.import_module("transformers")

    try:
        model_cls = getattr(transformers, entry["model_cls"])
        processor_cls = getattr(transformers, entry["processor_cls"])
    except AttributeError as exc:
        raise ImportError(
            f"Could not find {entry['model_cls']} / {entry['processor_cls']} "
            f"in your transformers installation.\n"
            f"Original error: {exc}\n\n"
            "Common fixes:\n"
            "  • Upgrade transformers: pip install 'transformers>=4.51.0'\n"
            "  • Install torchvision (required by the image processor):\n"
            "      pip install torchvision\n"
            "    For CUDA builds: pip install torchvision "
            "--index-url https://download.pytorch.org/whl/cu121"
        ) from exc

    processor = processor_cls.from_pretrained(model_dir)
    model = model_cls.from_pretrained(model_dir, device_map=device)
    return model, processor


class HFLayoutDetector:
    """Transformer-based document-layout detector for PP-DocLayoutV2/V3.

    A single forward pass is used per ``__call__``, with batch size equal to
    ``len(images)``. The ``batch_size`` field in YAML mainly affects pipeline
    queue batching, not this detector's internal tensor batching.

    Exposes the same callable interface as the PaddlePaddle
    ``LayoutAnalysisPredictor`` so it is a drop-in replacement inside
    :class:`~paddlex.inference.pipelines.paddleocr_vl.pipeline._PaddleOCRVLPipeline`.

    Each call yields one result dict per input image::

        {
            "input_path": None,
            "page_index": None,
            "boxes": [
                {
                    "cls_id":     int,
                    "label":      str,
                    "score":      float,
                    "coordinate": [xmin, ymin, xmax, ymax],
                    "order":      int,
                    # Only present when model_version="v3":
                    "polygon_points": [[x, y], ...],
                },
                ...
            ],
        }

    Parameters
    ----------
    model_dir:
        Local directory **or** HuggingFace Hub repo id.
        V2 example: ``"PaddlePaddle/PP-DocLayoutV2_safetensors"``
        V3 example: ``"PaddlePaddle/PP-DocLayoutV3_safetensors"``
    model_version:
        ``"v2"`` or ``"v3"``.  When ``None`` (default) the version is
        auto-detected from ``config.json``'s ``architectures`` field.
    device:
        ``"cpu"``, ``"gpu"``, ``"gpu:0"``, ``"cuda"``, ``"cuda:0"``.
    batch_size:
        Exposed as ``batch_sampler.batch_size`` for queue-mode sizing.
    threshold:
        Default confidence threshold.
    layout_nms:
        Apply class-agnostic IoU NMS after model post-processing.
    layout_nms_iou_threshold:
        IoU threshold for built-in NMS.
    """

    def __init__(
        self,
        model_dir: str,
        model_version: Optional[str] = None,
        device: Optional[str] = "cpu",
        batch_size: int = 1,
        threshold: float = 0.5,
        layout_nms: bool = True,
        layout_nms_iou_threshold: float = 0.5,
    ) -> None:
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "HFLayoutDetector requires 'torch'. "
                "Install it from https://pytorch.org/get-started/locally/"
            ) from exc

        self._torch = torch
        self.threshold = threshold
        self.default_layout_nms = layout_nms
        self.layout_nms_iou_threshold = layout_nms_iou_threshold
        self.batch_sampler = _BatchSampler(batch_size)
        self._device = self._parse_device(device)

        # Resolve model version
        if model_version is None:
            model_version = _detect_version_from_config(model_dir)
        self.model_version = model_version.lower()

        self._model, self._processor = _load_model_and_processor(
            model_dir, self.model_version, device=self._device
        )
        self._model.eval()
        self._id2label: Dict[int, str] = self._model.config.id2label

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def __call__(
        self,
        images: List[np.ndarray],
        threshold: Optional[float] = None,
        layout_nms: Optional[bool] = None,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]] = None,
        layout_merge_bboxes_mode: Optional[str] = None,
        layout_shape_mode: Optional[str] = "auto",
        filter_overlap_boxes: bool = False,
    ) -> Iterator[Dict]:
        """Detect layout regions; yield one result dict per input image.

        Parameters
        ----------
        images:
            BGR numpy arrays (OpenCV convention).
        threshold:
            Per-call confidence threshold (overrides ``__init__`` value).
        layout_nms:
            Per-call NMS toggle (overrides ``__init__`` value).
        layout_unclip_ratio:
            Expand boxes outward.  Scalar, ``(w_ratio, h_ratio)`` tuple, or
            per-label dict (dict falls back to no expansion at this level).
        layout_merge_bboxes_mode:
            Accepted for API compatibility; not used here.
        layout_shape_mode:
            ``"rect"`` suppresses polygon output even for V3.  Any other value
            causes V3 to include ``polygon_points`` in each box dict.
        filter_overlap_boxes:
            Accepted for API compatibility; overlap filtering is done later
            by the pipeline itself.
        """
        import torch

        score_threshold = threshold if threshold is not None else self.threshold
        apply_nms = layout_nms if layout_nms is not None else self.default_layout_nms
        want_polygons = (
            self.model_version == "v3" and layout_shape_mode != "rect"
        )

        if not images:
            return

        import torch
        from PIL import Image as _PIL_Image

        pil_list = []
        hw_list: List[Tuple[int, int]] = []
        for img_bgr in images:
            h, w = img_bgr.shape[:2]
            hw_list.append((h, w))
            pil_list.append(_PIL_Image.fromarray(img_bgr[:, :, ::-1]))

        # One forward pass with batch size == len(images) (matches request size).
        # _GPU_LOCK serialises CUDA work across pool threads so cuDNN
        # batch-norm doesn't collide (CUDNN_STATUS_NOT_SUPPORTED… error).
        with _GPU_LOCK:
            inputs = self._processor(images=pil_list, return_tensors="pt")
            inputs = {k: v.to(self._device) for k, v in inputs.items()}

            with torch.no_grad():
                outputs = self._model(**inputs)

            target_sizes = torch.tensor(
                [[h, w] for h, w in hw_list],
                device=self._device,
                dtype=torch.long,
            )
            raw_list = self._processor.post_process_object_detection(
                outputs,
                threshold=score_threshold,
                target_sizes=target_sizes,
            )
        # Some processor versions return a single dict when batch_size == 1.
        if isinstance(raw_list, dict):
            raw_list = [raw_list]

        for result, (h, w) in zip(raw_list, hw_list):
            yield self._detection_result_to_page(
                result,
                h,
                w,
                apply_nms=apply_nms,
                layout_unclip_ratio=layout_unclip_ratio,
                want_polygons=want_polygons,
            )

    def _detection_result_to_page(
        self,
        result: Dict,
        h: int,
        w: int,
        *,
        apply_nms: bool,
        layout_unclip_ratio: Optional[Union[float, Tuple[float, float], dict]],
        want_polygons: bool,
    ) -> Dict:
        """Turn one ``post_process_object_detection`` output dict into page res."""
        scores = result["scores"].cpu().numpy()
        label_ids = result["labels"].cpu().numpy()
        boxes = result["boxes"].cpu().numpy()

        polygon_points_list: Optional[List] = None
        if want_polygons and "polygon_points" in result:
            polygon_points_list = result["polygon_points"]

        if apply_nms and len(boxes) > 0:
            keep = self._nms(boxes, scores, self.layout_nms_iou_threshold)
            scores = scores[keep]
            label_ids = label_ids[keep]
            boxes = boxes[keep]
            if polygon_points_list is not None:
                polygon_points_list = [polygon_points_list[i] for i in keep]

        if layout_unclip_ratio is not None and len(boxes) > 0:
            boxes = self._unclip_boxes(boxes, layout_unclip_ratio, w, h)

        box_list: List[Dict] = []
        for idx, (score, label_id, box) in enumerate(
            zip(scores, label_ids, boxes)
        ):
            label_name = self._id2label.get(int(label_id), str(label_id))
            x1, y1, x2, y2 = box
            x1 = int(max(0, x1))
            y1 = int(max(0, y1))
            x2 = int(min(w, x2))
            y2 = int(min(h, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            entry: Dict = {
                "cls_id": int(label_id),
                "label": label_name,
                "score": float(score),
                "coordinate": [x1, y1, x2, y2],
                "order": idx + 1,
            }
            if polygon_points_list is not None:
                poly = polygon_points_list[idx]
                if poly is not None:
                    if hasattr(poly, "tolist"):
                        poly = poly.tolist()
                    entry["polygon_points"] = poly
            box_list.append(entry)

        return {
            "input_path": None,
            "page_index": None,
            "boxes": box_list,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_device(device: Optional[str]) -> str:
        """Normalise PaddlePaddle / CUDA device strings to torch format."""
        if not device or device == "cpu":
            return "cpu"
        if device.startswith("gpu"):
            parts = device.split(":", 1)
            return f"cuda:{parts[1]}" if len(parts) > 1 else "cuda"
        return device

    @staticmethod
    def _nms(
        boxes: np.ndarray,
        scores: np.ndarray,
        iou_threshold: float = 0.5,
    ) -> np.ndarray:
        """Class-agnostic IoU NMS; returns kept indices sorted by score."""
        if boxes.size == 0:
            return np.array([], dtype=np.int64)
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)
        order = scores.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0:
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(x1[i], x1[rest])
            yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest])
            yy2 = np.minimum(y2[i], y2[rest])
            inter = np.maximum(0.0, xx2 - xx1) * np.maximum(0.0, yy2 - yy1)
            iou = inter / (areas[i] + areas[rest] - inter + 1e-10)
            order = rest[iou <= iou_threshold]
        return np.array(keep, dtype=np.int64)

    @staticmethod
    def _unclip_boxes(
        boxes: np.ndarray,
        ratio: Union[float, Tuple[float, float], dict],
        img_w: int,
        img_h: int,
    ) -> np.ndarray:
        """Expand boxes outward by *ratio*."""
        if isinstance(ratio, dict):
            return boxes  # per-label unclipping not supported at this level
        if isinstance(ratio, (list, tuple)) and len(ratio) == 2:
            ratio_w, ratio_h = float(ratio[0]), float(ratio[1])
        else:
            ratio_w = ratio_h = float(ratio)

        boxes = boxes.copy().astype(float)
        cx = (boxes[:, 0] + boxes[:, 2]) / 2
        cy = (boxes[:, 1] + boxes[:, 3]) / 2
        bw = (boxes[:, 2] - boxes[:, 0]) * ratio_w
        bh = (boxes[:, 3] - boxes[:, 1]) * ratio_h
        boxes[:, 0] = np.clip(cx - bw / 2, 0, img_w)
        boxes[:, 1] = np.clip(cy - bh / 2, 0, img_h)
        boxes[:, 2] = np.clip(cx + bw / 2, 0, img_w)
        boxes[:, 3] = np.clip(cy + bh / 2, 0, img_h)
        return boxes
