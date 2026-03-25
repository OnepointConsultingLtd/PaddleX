# PaddleOCR-VL HuggingFace server (hf_server.py) on NVIDIA vLLM base image.
#
# Base image: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm
# Tag 26.02-py3 ships Python, CUDA, PyTorch, and vLLM-compatible stacks.
# Override at build time: docker build --build-arg BASE_IMAGE=nvcr.io/nvidia/vllm:25.12-py3 ...
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Base image includes torch; pin transformers/accelerate for PP-DocLayout + Qwen2-VL style models.
# torchvision: required by PP-DocLayoutV3 image processors in transformers.
RUN pip install --no-cache-dir \
    "transformers>=5.3" \
    "accelerate>=0.26" \
    "filetype>=1.2" \
    "torchvision"  \
    "openai"

# Install PaddleX from build context (repository root).
COPY . /app/PaddleX
WORKDIR /app/PaddleX

RUN pip install --no-cache-dir -e ".[ocr,ocr-hf,serving]"

EXPOSE 8080

# Default: single worker; increase pool or point config at vLLM backend via env override.
ENV PADDLEX_HF_CONFIG=paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml

CMD ["sh", "-c", "python hf_server.py --config \"${PADDLEX_HF_CONFIG}\" --host 0.0.0.0 --port 8080 --pool-size 2"]
