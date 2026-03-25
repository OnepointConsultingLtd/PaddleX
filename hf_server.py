#!/usr/bin/env python3
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
"""PaddleOCR-VL pool inference server.

Maintains a configurable pool of pipeline instances and routes each request to
the least-busy pipeline instance, then returns the pipeline output.

Reuses the existing PaddleX serving infrastructure:
  - paddlex.inference.serving.basic_serving._app:
      PipelineWrapper, AppContext, primary_operation
  - paddlex.inference.serving.basic_serving._server: run_server
  - paddlex.inference.serving.schemas.paddleocr_vl: all request/response schemas
  - paddlex.inference.serving.infra: utils, config, models
  - paddlex.inference.serving.basic_serving._pipeline_apps._common: ocr, common

Usage:
  python hf_server.py \\
      --config paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml \\
      --pool-size 2 \\
      --host 0.0.0.0 \\
      --port 8080

  # Override device for all instances:
  python hf_server.py --config ... --pool-size 2 --device cuda:0

  # Layout-only (boxes + labels, no VLM), POST /layout-detection with JSON body
  # {"images": ["<base64>", ...], "layoutShapeMode": "auto"}.

Dependencies (in addition to paddlex core):
  pip install fastapi uvicorn aiohttp
"""

import argparse
import asyncio
import base64
import contextlib
import copy
import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Tuple, Union

from pydantic import BaseModel, Field
from typing_extensions import Literal

import aiohttp
import fastapi
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException

from paddlex import create_pipeline
from paddlex.inference.serving.basic_serving._app import (
    AppContext,
    PipelineWrapper,
    primary_operation,
)
from paddlex.inference.serving.basic_serving._pipeline_apps._common import common
from paddlex.inference.serving.basic_serving._pipeline_apps._common import (
    ocr as ocr_common,
)
from paddlex.inference.serving.basic_serving._server import run_server
from paddlex.inference.serving.infra import utils as serving_utils
from paddlex.inference.serving.infra.config import AppConfig
from paddlex.inference.serving.infra.models import (
    AIStudioNoResultResponse,
    AIStudioResultResponse,
)
from paddlex.inference.serving.schemas.paddleocr_vl import (
    INFER_ENDPOINT,
    RESTRUCTURE_PAGES_ENDPOINT,
    InferRequest,
    InferResult,
    RestructurePagesRequest,
    RestructurePagesResult,
)

LAYOUT_DETECTION_ONLY_ENDPOINT = "/layout-detection"


class LayoutDetectionOnlyRequest(BaseModel):
    """Base64-encoded image pages (same encoding style as ``file`` in infer)."""

    images: List[str] = Field(
        ...,
        min_length=1,
        description="Each entry: raw base64 or data URL (data:image/...;base64,...).",
    )
    logId: Optional[str] = None
    layoutThreshold: Optional[float] = None
    layoutNms: Optional[bool] = None
    layoutUnclipRatio: Optional[Union[float, Tuple[float, float], dict]] = None
    layoutShapeMode: Literal["rect", "quad", "poly", "auto"] = "auto"


class LayoutPageOut(BaseModel):
    pageIndex: int
    width: int
    height: int
    boxes: List[Dict[str, Any]]


class LayoutDetectionOnlyResult(BaseModel):
    pages: List[LayoutPageOut]


def _decode_base64_image_payload(s: str) -> bytes:
    if "," in s and s.strip().startswith("data:"):
        s = s.split(",", 1)[1]
    return base64.b64decode(s)


# ---------------------------------------------------------------------------
# Pipeline pool
# ---------------------------------------------------------------------------


class PipelinePool:
    """Routes inference requests to the least-busy PipelineWrapper.

    Implements the same async interface as PipelineWrapper (infer / call /
    close / .pipeline property) so it is completely transparent to the route
    handlers that were written for a single pipeline.

    Least-busy selection is based on the number of pending items in each
    wrapper's internal Queue (PipelineWrapper._queue.qsize()).
    """

    def __init__(self, wrappers: List[PipelineWrapper]) -> None:
        if not wrappers:
            raise ValueError("PipelinePool requires at least one wrapper")
        self._wrappers = wrappers

    @property
    def pipeline(self) -> Any:
        """Return the underlying pipeline object of the first instance.

        Used for CPU-only operations like restructure_pages that are
        stateless and don't require GPU resources.
        """
        return self._wrappers[0].pipeline

    def _least_busy(self) -> PipelineWrapper:
        """Pick the wrapper with the fewest pending jobs."""
        return min(self._wrappers, key=lambda w: w._queue.qsize())

    async def infer(self, *args: Any, **kwargs: Any) -> list:
        """Delegate inference to the least-busy pipeline instance."""
        return await self._least_busy().infer(*args, **kwargs)

    async def call(self, func: Any, *args: Any, **kwargs: Any) -> Any:
        """Delegate an arbitrary call to the least-busy pipeline instance."""
        return await self._least_busy().call(func, *args, **kwargs)

    async def close(self) -> None:
        """Close all pipeline instances in the pool."""
        for w in self._wrappers:
            await w.close()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_pool_app(
    pipelines: list,
    app_config: AppConfig,
) -> "fastapi.FastAPI":
    """Create a FastAPI application backed by a pool of pipeline instances.

    The pool lifespan:
      1. Wraps each pipeline in a PipelineWrapper (dedicated inference thread,
         same pattern as the existing basic_serving infrastructure).
      2. Assembles a PipelinePool that routes to the least-busy wrapper.
      3. Opens a shared aiohttp.ClientSession for async HTTP file fetching.

    Returns a FastAPI app with /health, /layout-parsing,
    /restructure-pages, and /layout-detection endpoints pre-registered.
    """

    @contextlib.asynccontextmanager
    async def _lifespan(app: "fastapi.FastAPI") -> AsyncGenerator[None, None]:
        wrappers = [PipelineWrapper(p) for p in pipelines]
        pool = PipelinePool(wrappers)
        ctx.pipeline = pool  # type: ignore[assignment]  – duck-typed as PipelineWrapper
        async with aiohttp.ClientSession(
            cookie_jar=aiohttp.DummyCookieJar()
        ) as aiohttp_session:
            ctx.aiohttp_session = aiohttp_session
            try:
                yield
            finally:
                await pool.close()

    app = fastapi.FastAPI(lifespan=_lifespan)
    ctx = AppContext(config=app_config)
    app.state.context = ctx

    # Populate ctx.extra (file_storage, return_img_urls, max_output_img_size, …)
    ocr_common.update_app_context(ctx)

    # ------------------------------------------------------------------
    # Health endpoint
    # ------------------------------------------------------------------

    @app.get("/health", operation_id="checkHealth")
    async def _health() -> AIStudioNoResultResponse:
        return AIStudioNoResultResponse(
            logId=serving_utils.generate_log_id(), errorCode=0, errorMsg="Healthy"
        )

    # ------------------------------------------------------------------
    # Exception handlers (mirrors _app.py create_app)
    # ------------------------------------------------------------------

    async def _try_log_id(request: fastapi.Request) -> Optional[str]:
        try:
            body = await request.json()
            if isinstance(body, dict):
                return body.get("logId")
        except Exception:
            pass
        return None

    def _loc_to_dot(loc: tuple) -> str:
        path = ""
        for i, x in enumerate(loc):
            if isinstance(x, str):
                if i > 0:
                    path += "."
                path += x
            elif isinstance(x, int):
                path += f"[{x}]"
        return path

    def _convert_errors(exc: Any) -> list:
        return [
            {"type": e["type"], "loc": _loc_to_dot(e["loc"]), "msg": e["msg"]}
            for e in exc.errors()
        ]

    @app.exception_handler(RequestValidationError)
    async def _validation_err(
        req: fastapi.Request, exc: RequestValidationError
    ) -> JSONResponse:
        lid = await _try_log_id(req) or serving_utils.generate_log_id()
        return JSONResponse(
            content=jsonable_encoder(
                AIStudioNoResultResponse(
                    logId=lid,
                    errorCode=422,
                    errorMsg=json.dumps(_convert_errors(exc)),
                )
            ),
            status_code=422,
        )

    @app.exception_handler(HTTPException)
    async def _http_err(req: fastapi.Request, exc: HTTPException) -> JSONResponse:
        lid = await _try_log_id(req) or serving_utils.generate_log_id()
        return JSONResponse(
            content=jsonable_encoder(
                AIStudioNoResultResponse(
                    logId=lid, errorCode=exc.status_code, errorMsg=exc.detail
                )
            ),
            status_code=exc.status_code,
        )

    @app.exception_handler(Exception)
    async def _unexpected_err(req: fastapi.Request, exc: Exception) -> JSONResponse:
        lid = await _try_log_id(req) or serving_utils.generate_log_id()
        logging.exception("Unhandled exception")
        return JSONResponse(
            content=jsonable_encoder(
                AIStudioNoResultResponse(
                    logId=lid, errorCode=500, errorMsg="Internal server error"
                )
            ),
            status_code=500,
        )

    # ------------------------------------------------------------------
    # PaddleOCR-VL routes (mirrors _pipeline_apps/paddleocr_vl.py)
    # ctx.pipeline is a PipelinePool but exposes the same interface as
    # PipelineWrapper, so the route logic is identical.
    # ------------------------------------------------------------------

    @primary_operation(app, INFER_ENDPOINT, "infer")
    async def _infer(
        request: InferRequest,
    ) -> AIStudioResultResponse[InferResult]:
        pipeline = ctx.pipeline
        log_id = request.logId or serving_utils.generate_log_id()
        visualize_enabled = (
            request.visualize if request.visualize is not None else ctx.config.visualize
        )
        images, data_info = await ocr_common.get_images(request, ctx)

        result = await pipeline.infer(
            images,
            use_doc_orientation_classify=request.useDocOrientationClassify,
            use_doc_unwarping=request.useDocUnwarping,
            use_layout_detection=request.useLayoutDetection,
            use_chart_recognition=request.useChartRecognition,
            use_seal_recognition=request.useSealRecognition,
            use_ocr_for_image_block=request.useOcrForImageBlock,
            layout_threshold=request.layoutThreshold,
            layout_nms=request.layoutNms,
            layout_unclip_ratio=request.layoutUnclipRatio,
            layout_merge_bboxes_mode=request.layoutMergeBboxesMode,
            layout_shape_mode=request.layoutShapeMode,
            prompt_label=request.promptLabel,
            format_block_content=request.formatBlockContent,
            repetition_penalty=request.repetitionPenalty,
            temperature=request.temperature,
            top_p=request.topP,
            min_pixels=request.minPixels,
            max_pixels=request.maxPixels,
            max_new_tokens=request.maxNewTokens,
            merge_layout_blocks=request.mergeLayoutBlocks,
            markdown_ignore_labels=request.markdownIgnoreLabels,
            vlm_extra_args=request.vlmExtraArgs,
        )

        if request.restructurePages:
            result = await serving_utils.call_async(
                pipeline.pipeline.restructure_pages,
                result,
                merge_tables=request.mergeTables,
                relevel_titles=request.relevelTitles,
                concatenate_pages=False,
            )
            result = list(result)

        layout_parsing_results = []
        for i, (img, item) in enumerate(zip(images, result)):
            pruned_res = common.prune_result(item.json["res"])
            md_data = item._to_markdown(
                pretty=request.prettifyMarkdown,
                show_formula_number=request.showFormulaNumber,
            )
            md_imgs = await serving_utils.call_async(
                common.postprocess_images,
                md_data["markdown_images"],
                log_id,
                filename_template=f"markdown_{i}/{{key}}",
                file_storage=ctx.extra["file_storage"],
                return_urls=ctx.extra["return_img_urls"],
                max_img_size=ctx.extra["max_output_img_size"],
            )
            imgs = {}
            if visualize_enabled:
                imgs = {"input_img": img, **item.img}
                imgs = await serving_utils.call_async(
                    common.postprocess_images,
                    imgs,
                    log_id,
                    filename_template=f"{{key}}_{i}.jpg",
                    file_storage=ctx.extra["file_storage"],
                    return_urls=ctx.extra["return_img_urls"],
                    max_img_size=ctx.extra["max_output_img_size"],
                )
            layout_parsing_results.append(
                dict(
                    prunedResult=pruned_res,
                    markdown=dict(
                        text=md_data["markdown_texts"],
                        images=md_imgs,
                    ),
                    outputImages=(
                        {k: v for k, v in imgs.items() if k != "input_img"}
                        if imgs
                        else None
                    ),
                    inputImage=imgs.get("input_img"),
                )
            )

        return AIStudioResultResponse[InferResult](
            logId=log_id,
            result=InferResult(
                layoutParsingResults=layout_parsing_results,
                dataInfo=data_info,
            ),
        )

    @primary_operation(app, RESTRUCTURE_PAGES_ENDPOINT, "restructurePages")
    async def _restructure_pages(
        request: RestructurePagesRequest,
    ) -> AIStudioResultResponse[RestructurePagesResult]:
        pipeline = ctx.pipeline
        log_id = request.logId or serving_utils.generate_log_id()

        original_results = [
            {"res": {**page.prunedResult, "input_path": "", "page_index": i}}
            for i, page in enumerate(request.pages)
        ]
        markdown_images: dict = {}
        if request.concatenatePages:
            for page in request.pages:
                markdown_images.update(page.markdownImages or {})

        restructured = await serving_utils.call_async(
            pipeline.pipeline.restructure_pages,
            original_results,
            merge_tables=request.mergeTables,
            relevel_titles=request.relevelTitles,
            concatenate_pages=request.concatenatePages,
        )
        restructured = list(restructured)

        layout_parsing_results = []
        if request.concatenatePages:
            md_data = restructured[0]._to_markdown(
                pretty=request.prettifyMarkdown,
                show_formula_number=request.showFormulaNumber,
            )
            layout_parsing_results.append(
                dict(
                    prunedResult=common.prune_result(restructured[0].json["res"]),
                    markdown=dict(
                        text=md_data["markdown_texts"],
                        images=markdown_images,
                    ),
                )
            )
        else:
            for new_res, old_page in zip(restructured, request.pages):
                md_data = new_res._to_markdown(
                    pretty=request.prettifyMarkdown,
                    show_formula_number=request.showFormulaNumber,
                )
                layout_parsing_results.append(
                    dict(
                        prunedResult=common.prune_result(new_res.json["res"]),
                        markdown=dict(
                            text=md_data["markdown_texts"],
                            images=old_page.markdownImages or {},
                        ),
                    )
                )

        return AIStudioResultResponse[RestructurePagesResult](
            logId=log_id,
            result=RestructurePagesResult(
                layoutParsingResults=layout_parsing_results,
            ),
        )

    @primary_operation(
        app,
        LAYOUT_DETECTION_ONLY_ENDPOINT,
        "layoutDetectionOnly",
    )
    async def _layout_detection_only(
        request: LayoutDetectionOnlyRequest,
    ) -> AIStudioResultResponse[LayoutDetectionOnlyResult]:
        """Layout only: bounding boxes + class labels, no VLM / OCR content."""
        log_id = request.logId or serving_utils.generate_log_id()
        pl0 = ctx.pipeline.pipeline
        if getattr(pl0, "layout_det_model", None) is None:
            raise HTTPException(
                status_code=503,
                detail="Layout detection is not enabled or not available in this pipeline",
            )
        max_n = int(ctx.extra.get("max_num_input_imgs", 10))
        if len(request.images) > max_n:
            raise HTTPException(
                status_code=422,
                detail=f"Too many images (maximum {max_n})",
            )

        imgs_bgr: List[Any] = []
        for b64 in request.images:
            try:
                raw = _decode_base64_image_payload(b64)
            except Exception as e:
                raise HTTPException(
                    status_code=422, detail=f"Invalid base64 image: {e}"
                ) from e
            arr = serving_utils.image_bytes_to_array(raw)
            if arr is None:
                raise HTTPException(status_code=422, detail="Could not decode image")
            imgs_bgr.append(arr)

        pipeline = ctx.pipeline

        def _layout_only() -> List[Dict[str, Any]]:
            pl = pipeline.pipeline
            det = getattr(pl, "layout_det_model", None)
            if det is None:
                raise RuntimeError("layout_det_model is not available in this pipeline")
            out = list(
                det(
                    imgs_bgr,
                    threshold=request.layoutThreshold,
                    layout_nms=request.layoutNms,
                    layout_unclip_ratio=request.layoutUnclipRatio,
                    layout_shape_mode=request.layoutShapeMode,
                )
            )
            return out

        raw_pages = await pipeline.call(_layout_only)

        pages_out: List[LayoutPageOut] = []
        for i, page in enumerate(raw_pages):
            h, w = imgs_bgr[i].shape[:2]
            pages_out.append(
                LayoutPageOut(
                    pageIndex=i,
                    width=int(w),
                    height=int(h),
                    boxes=page["boxes"],
                )
            )

        return AIStudioResultResponse[LayoutDetectionOnlyResult](
            logId=log_id,
            result=LayoutDetectionOnlyResult(pages=pages_out),
        )

    return app


def _get_inner_pipeline(pipeline: Any) -> Any:
    """Reach the inner _PaddleOCRVLPipeline through the AutoParallel* wrapper."""
    if hasattr(pipeline, "_pipeline"):
        return pipeline._pipeline
    return pipeline


def _load_pipelines(
    config_path: str, pool_size: int, device: Optional[str]
) -> list:
    """Load *pool_size* pipeline instances, sharing one layout detector.

    The first pipeline is loaded normally (including the GPU layout model).
    Subsequent pipelines receive a config copy with ``use_layout_detection``
    set to ``False`` so they skip loading their own detector.  After loading,
    the first pipeline's ``layout_det_model`` is injected into every instance.

    This avoids loading N copies of PP-DocLayoutV3 on the same GPU, which
    exhausts cuDNN workspace and causes
    ``CUDNN_STATUS_NOT_SUPPORTED_SUBLIBRARY_UNAVAILABLE``.
    """
    from paddlex.inference.pipelines import load_pipeline_config

    base_config = load_pipeline_config(config_path)

    print(f"  Loading instance 1/{pool_size} (with layout detector)...")
    first = create_pipeline(config=copy.deepcopy(base_config), device=device)
    pipelines = [first]

    if pool_size > 1:
        inner_first = _get_inner_pipeline(first)
        shared_det = getattr(inner_first, "layout_det_model", None)
        use_ld = getattr(inner_first, "use_layout_detection", True)

        for i in range(1, pool_size):
            print(f"  Loading instance {i + 1}/{pool_size} (layout detector shared)...")
            cfg = copy.deepcopy(base_config)
            cfg["use_layout_detection"] = False
            p = create_pipeline(config=cfg, device=device)
            inner = _get_inner_pipeline(p)
            if shared_det is not None:
                inner.layout_det_model = shared_det
                inner.use_layout_detection = use_ld
            pipelines.append(p)

        if shared_det is not None:
            print(f"  Shared layout detector across {pool_size} pool instance(s)")

    return pipelines


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PaddleOCR-VL pool inference server",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        default="paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml",
        help="Path to the pipeline YAML config file",
    )
    parser.add_argument(
        "--pool-size",
        type=int,
        default=1,
        help="Number of pipeline instances to maintain in the pool",
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host address to bind",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Override device for all pipeline instances (e.g. 'gpu:0', 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        default=False,
        help="Disable visualization images in responses (reduces response size)",
    )
    args = parser.parse_args()

    print(
        f"Loading {args.pool_size} pipeline instance(s) from: {args.config}"
        + (f"  [device={args.device}]" if args.device else "")
    )

    pipelines = _load_pipelines(args.config, args.pool_size, args.device)

    print(
        f"All {args.pool_size} pipeline instance(s) loaded. "
        f"Starting server on http://{args.host}:{args.port}"
    )
    print(f"  POST {INFER_ENDPOINT}          — run inference")
    print(f"  POST {RESTRUCTURE_PAGES_ENDPOINT}  — restructure pages")
    print(f"  POST {LAYOUT_DETECTION_ONLY_ENDPOINT}  — layout boxes only (batched)")
    print(f"  GET  /health                    — health check")

    app_config = AppConfig(visualize=not args.no_visualize)
    app = create_pool_app(pipelines, app_config)
    run_server(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
