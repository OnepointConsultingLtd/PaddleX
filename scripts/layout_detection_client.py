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

"""Batch client for hf_server ``POST /layout-detection``.

Reads images from a directory, sends them in batches to the server, writes one
JSON file per image (same relative path, ``.json`` extension). Uses only the
stdlib (no ``requests`` / ``aiohttp`` required).

Example::

    python scripts/layout_detection_client.py \\
        --base-url http://127.0.0.1:8080 \\
        --input-dir ./pages \\
        --output-dir ./out_layouts \\
        --images-per-request 4 \\
        --parallel 2
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


@dataclass
class BatchResult:
    batch_index: int
    paths: List[Path]
    ok: bool
    elapsed_s: float
    error: Optional[str] = None
    http_status: Optional[int] = None


@dataclass
class Totals:
    images_ok: int = 0
    images_failed: int = 0
    batches_ok: int = 0
    batches_failed: int = 0
    wall_s: float = 0.0
    sum_batch_wall_s: float = 0.0  # sum of per-batch durations (for avg batch time)


def _normalize_base_url(url: str) -> str:
    return url.rstrip("/")


def _collect_images(input_dir: Path, recursive: bool) -> List[Path]:
    input_dir = input_dir.resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {input_dir}")
    out: List[Path] = []
    if recursive:
        for p in sorted(input_dir.rglob("*")):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                out.append(p)
    else:
        for p in sorted(input_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES:
                out.append(p)
    return out


def _b64_file(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def _post_layout_detection(
    endpoint: str, image_paths: List[Path], timeout_s: float
) -> Tuple[Dict[str, Any], int]:
    payload = {"images": [_b64_file(p) for p in image_paths]}
    data = json.dumps(payload).encode("utf-8")
    req = Request(
        endpoint,
        data=data,
        method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    with urlopen(req, timeout=timeout_s) as resp:
        body = resp.read().decode("utf-8")
        status = resp.status
    return json.loads(body), status


def _write_page_json(
    output_root: Path,
    input_root: Path,
    source_file: Path,
    page: Dict[str, Any],
) -> None:
    rel = source_file.resolve().relative_to(input_root.resolve())
    out_path = output_root / rel.with_suffix(".json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    doc = {
        "sourceFile": str(rel).replace("\\", "/"),
        "pageIndex": page.get("pageIndex"),
        "width": page.get("width"),
        "height": page.get("height"),
        "boxes": page.get("boxes", []),
    }
    out_path.write_text(json.dumps(doc, indent=2, ensure_ascii=False), encoding="utf-8")


def _process_batch(
    batch_index: int,
    paths: List[Path],
    endpoint: str,
    output_dir: Path,
    input_root: Path,
    timeout_s: float,
    print_lock: threading.Lock,
) -> BatchResult:
    t0 = time.perf_counter()
    if not paths:
        return BatchResult(batch_index, paths, True, 0.0)

    try:
        body, status = _post_layout_detection(endpoint, paths, timeout_s)
    except HTTPError as e:
        err = e.read().decode("utf-8", errors="replace")[:500]
        elapsed = time.perf_counter() - t0
        with print_lock:
            print(
                f"[batch {batch_index + 1}] HTTP {e.code}  {elapsed:.2f}s  FAIL  {err[:200]}",
                file=sys.stderr,
            )
        return BatchResult(
            batch_index, paths, False, elapsed, error=f"HTTP {e.code}: {err}", http_status=e.code
        )
    except (URLError, OSError, json.JSONDecodeError, ValueError) as e:
        elapsed = time.perf_counter() - t0
        with print_lock:
            print(f"[batch {batch_index + 1}] error  {elapsed:.2f}s  FAIL  {e}", file=sys.stderr)
        return BatchResult(batch_index, paths, False, elapsed, error=str(e))

    elapsed = time.perf_counter() - t0
    err_code = body.get("errorCode", 0)
    if err_code != 0:
        msg = body.get("errorMsg", str(body))
        with print_lock:
            print(
                f"[batch {batch_index + 1}] API errorCode={err_code}  {elapsed:.2f}s  FAIL  {msg[:200]}",
                file=sys.stderr,
            )
        return BatchResult(
            batch_index, paths, False, elapsed, error=f"errorCode={err_code}: {msg}"
        )

    result = body.get("result") or {}
    pages = result.get("pages")
    if not isinstance(pages, list) or len(pages) != len(paths):
        with print_lock:
            print(
                f"[batch {batch_index + 1}] bad response: expected {len(paths)} pages, got {type(pages)}",
                file=sys.stderr,
            )
        return BatchResult(
            batch_index,
            paths,
            False,
            elapsed,
            error=f"page count mismatch: {len(pages) if isinstance(pages, list) else pages}",
        )

    for pth, page in zip(paths, pages):
        _write_page_json(output_dir, input_root, pth, page)

    n = len(paths)
    ips = n / elapsed if elapsed > 0 else 0.0
    with print_lock:
        print(
            f"[batch {batch_index + 1}]  {n} img  {elapsed:.2f}s  {ips:.2f} img/s  OK",
            flush=True,
        )

    return BatchResult(batch_index, paths, True, elapsed, http_status=status)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Send images to /layout-detection and write one JSON per file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--base-url",
        required=True,
        help="Server base URL (e.g. http://127.0.0.1:8080)",
    )
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory of images")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for JSON outputs (mirrors relative paths when using --recursive)",
    )
    parser.add_argument(
        "--images-per-request",
        type=int,
        default=1,
        help="Number of images per HTTP request (server batches layout model)",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=1,
        help="Maximum concurrent HTTP requests",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
        help="Include images in subdirectories of --input-dir",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=600.0,
        help="Per-request timeout in seconds",
    )
    args = parser.parse_args()

    if args.images_per_request < 1:
        print("--images-per-request must be >= 1", file=sys.stderr)
        return 2
    if args.parallel < 1:
        print("--parallel must be >= 1", file=sys.stderr)
        return 2

    try:
        images = _collect_images(args.input_dir, args.recursive)
    except FileNotFoundError as e:
        print(e, file=sys.stderr)
        return 2

    if not images:
        print("No image files found.", file=sys.stderr)
        return 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    base = _normalize_base_url(args.base_url)
    endpoint = f"{base}/layout-detection"

    batches: List[List[Path]] = []
    for i in range(0, len(images), args.images_per_request):
        batches.append(images[i : i + args.images_per_request])

    print(
        f"Server: {endpoint}\n"
        f"Images: {len(images)}  batches: {len(batches)}  "
        f"per-request: {args.images_per_request}  parallel: {args.parallel}\n",
        flush=True,
    )

    input_root = args.input_dir.resolve()
    print_lock = threading.Lock()
    totals = Totals()
    t_wall0 = time.perf_counter()

    with ThreadPoolExecutor(max_workers=args.parallel) as ex:
        futures = {
            ex.submit(
                _process_batch,
                bi,
                batch_paths,
                endpoint,
                args.output_dir,
                input_root,
                args.timeout,
                print_lock,
            ): bi
            for bi, batch_paths in enumerate(batches)
        }
        for fut in as_completed(futures):
            br = fut.result()
            totals.sum_batch_wall_s += br.elapsed_s
            if br.ok:
                totals.batches_ok += 1
                totals.images_ok += len(br.paths)
            else:
                totals.batches_failed += 1
                totals.images_failed += len(br.paths)

    totals.wall_s = time.perf_counter() - t_wall0

    overall_ips = totals.images_ok / totals.wall_s if totals.wall_s > 0 else 0.0
    avg_batch = totals.sum_batch_wall_s / len(batches) if batches else 0.0

    print(
        "\n--- Summary ---\n"
        f"  Images written (OK): {totals.images_ok}\n"
        f"  Images failed (batch errors): {totals.images_failed}\n"
        f"  Batches OK / failed: {totals.batches_ok} / {totals.batches_failed}\n"
        f"  Wall time: {totals.wall_s:.2f} s\n"
        f"  Throughput (images / wall time): {overall_ips:.2f} img/s\n"
        f"  Avg batch duration (sum/n batches, not wall): {avg_batch:.2f} s\n",
        flush=True,
    )

    return 0 if totals.batches_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
