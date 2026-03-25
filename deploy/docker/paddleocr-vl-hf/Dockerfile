# PaddleOCR-VL HuggingFace server (hf_server.py) on NVIDIA vLLM base image.
#
# Uses Astral uv (venv + uv pip install) — matches local workflows where plain
# pip against the base image fails; uv resolves wheels reliably.
#
# Base image: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
# Override: docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/vllm:25.12-py3 ...
#
# Build (from repository root):
#   docker build -f deploy/docker/paddleocr-vl-hf/Dockerfile -t paddlex-paddleocr-vl-hf:latest .
#
# Run (GPU):
#   docker run --gpus all --rm -p 8080:8080 \
#     -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true \
#     -v ~/.cache/huggingface:/root/.cache/huggingface \
#     paddlex-paddleocr-vl-hf:latest

ARG BASE_IMAGE=nvcr.io/nvidia/vllm:26.02-py3
FROM ${BASE_IMAGE}

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=1

# OpenCV (opencv-contrib-python) needs libGL at runtime for some code paths.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uv CLI (same tool as `uv venv` / `uv pip install` on the host).
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"

WORKDIR /app/PaddleX
COPY . /app/PaddleX

# Match working host sequence: uv venv → uv pip install … → editable paddlex.
RUN uv venv .venv
ENV VIRTUAL_ENV=/app/PaddleX/.venv \
    PATH="/app/PaddleX/.venv/bin:${PATH}"

RUN uv pip install "transformers>=4.50.0,<6" \
    && uv pip install "torchvision" \
    && uv pip install "accelerate>=0.26" \
    && uv pip install "filetype>=1.2" \
    && uv pip install --no-cache-dir -e ".[ocr,ocr-hf,serving]" \
    && uv pip install "openai"

EXPOSE 8080

# Default: vLLM backend config + pool of 2 (override with PADDLEX_HF_CONFIG).
ENV PADDLEX_HF_CONFIG=paddlex/configs/pipelines/PaddleOCR-VL-HF-vllm.yaml

CMD ["sh", "-c", "python hf_server.py --config \"${PADDLEX_HF_CONFIG}\" --host 0.0.0.0 --port 8080 --pool-size 2"]
