# PaddleOCR-VL (HuggingFace) — install, `hf_server`, Docker

This guide covers installing PaddleX with the **HuggingFace backends** (no PaddlePaddle for layout/VLM inference), running the **`hf_server.py`** HTTP service, and **Docker** on Linux (x86_64 GPU), **Windows**, and **Linux ARM64**.

## What gets installed

| Extra | Purpose |
|-------|---------|
| `ocr` | OpenCV, PDF, and OCR-related deps used by the pipeline |
| `ocr-hf` | `torch`, `transformers`, `accelerate`, `einops` for HF models |
| `serving` | `fastapi`, `uvicorn`, `aiohttp`, … for `hf_server.py` |

Install from a clone of this repository:

```bash
pip install -e ".[ocr,ocr-hf,serving]"
```

Python **3.10+** is recommended (match your PyTorch wheels).

---

## Windows (x86_64)

### 1. Prerequisites

- [Python 3.10+](https://www.python.org/downloads/) (64-bit)
- Optional: [CUDA](https://pytorch.org/get-started/locally/) if you use GPU PyTorch

### 2. Virtual environment

PowerShell:

```powershell
cd C:\path\to\PaddleX
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
```

Using **uv** (optional):

```powershell
pip install uv
uv venv
.\.venv\Scripts\Activate.ps1
uv pip install -e ".[ocr,ocr-hf,serving]"
```

If you use plain `pip` without `uv`:

```powershell
pip install -e ".[ocr,ocr-hf,serving]"
```

### 3. PyTorch / torchvision

Install builds that match your machine from [pytorch.org](https://pytorch.org/get-started/locally/). **torchvision** is required for PP-DocLayout V2/V3 image processors.

Example (CPU-only):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### 4. Optional: skip model-host connectivity check

```powershell
$env:PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK = "true"
```

### 5. Run the server

```powershell
python hf_server.py --config paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml --host 0.0.0.0 --port 8080 --pool-size 1
```

### 6. Test

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8080/health
```

---

## Linux (x86_64, native)

### 1. System packages (OpenCV runtime)

Debian / Ubuntu:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential libgl1 libglib2.0-0
```

### 2. venv and PaddleX

```bash
cd /path/to/PaddleX
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[ocr,ocr-hf,serving]"
```

### 3. PyTorch + torchvision

Pick CUDA or CPU from [pytorch.org](https://pytorch.org/get-started/locally/), e.g.:

```bash
# Example: CUDA 12.x (adjust index URL to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 4. Environment and server

```bash
export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true
python hf_server.py --config paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml --host 0.0.0.0 --port 8080 --pool-size 1
```

### 5. Test

```bash
curl -s http://127.0.0.1:8080/health
```

---

## Linux ARM64 (aarch64) — Raspberry Pi, AWS Graviton, Jetson (Linux), etc.

The **NVIDIA vLLM GPU image** used in the Docker section below is **typically amd64-only**. On ARM64, install **natively** with CPU (or Jetson/L4T-specific) PyTorch wheels.

### 1. System packages

```bash
sudo apt-get update
sudo apt-get install -y python3-venv python3-dev build-essential libgl1 libglib2.0-0
```

### 2. PyTorch for aarch64

Use the official instructions for your platform:

- **CPU wheels**: often `pip install torch torchvision` from PyPI if available for your Python + platform.
- **Jetson**: use NVIDIA’s [Jetson Zoo / PyTorch wheels](https://forums.developer.nvidia.com/) for your JetPack version — do **not** assume the same `pip` index as x86_64 CUDA.

### 3. PaddleX

```bash
cd /path/to/PaddleX
python3 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel
pip install -e ".[ocr,ocr-hf,serving]"
```

### 4. Run and test

Same as Linux x86_64 (`hf_server.py` + `curl` health check). Expect layout/VLM workloads to be **much slower** on CPU-only ARM unless you use hardware-optimized PyTorch.

---

## Docker (Linux x86_64 + NVIDIA GPU)

The Dockerfile uses **`nvcr.io/nvidia/vllm:26.02-py3`** (NVIDIA NGC “vLLM” container). The base image ships CUDA-capable **PyTorch**; dependency installation uses **[Astral uv](https://docs.astral.sh/uv/)** (`uv venv` + `uv pip install …`) so resolution matches a typical **uv** workflow on the host (plain `pip` in Docker often mis-resolves against this stack).

Installed inside the image (in order): `transformers`, `torchvision`, `accelerate`, `filetype`, editable **`paddlex[ocr,ocr-hf,serving]`**, and **`openai`** (for the vLLM / genai client path).

**Defaults:** `PADDLEX_HF_CONFIG=paddlex/configs/pipelines/PaddleOCR-VL-HF-vllm.yaml`, **`pool-size 2`**. Override `PADDLEX_HF_CONFIG` for the full in-process HF VLM config (`PaddleOCR-VL-HF.yaml`).

### Prerequisites

- [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) (`nvidia-container-toolkit`)
- Docker with GPU support (`docker run --gpus all ...`)

### Build (from repository root)

```bash
cd /path/to/PaddleX
docker build -f deploy/docker/paddleocr-vl-hf/Dockerfile -t paddlex-paddleocr-vl-hf:latest .
```

Optional: use another NGC tag (must match your driver/CUDA):

```bash
docker build -f deploy/docker/paddleocr-vl-hf/Dockerfile \
  --build-arg BASE_IMAGE=nvcr.io/nvidia/vllm:25.12-py3 \
  -t paddlex-paddleocr-vl-hf:latest .
```

### Run

Mount the Hugging Face cache so models download once:

```bash
docker run --gpus all --rm -p 8080:8080 \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  paddlex-paddleocr-vl-hf:latest
```

The default image already uses **`PaddleOCR-VL-HF-vllm.yaml`**. To switch to the full HuggingFace in-process VLM (`PaddleOCR-VL-HF.yaml`):

```bash
docker run --gpus all --rm -p 8080:8080 \
  -e PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=true \
  -e PADDLEX_HF_CONFIG=paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  paddlex-paddleocr-vl-hf:latest
```

For vLLM on the host, point `genai_config.server_url` in `PaddleOCR-VL-HF-vllm.yaml` at reachable hostnames (e.g. `--add-host=host.docker.internal:host-gateway` on Docker Desktop).

### Docker health test

```bash
curl -s http://127.0.0.1:8080/health
```

### Docker layout-only endpoint test

Two base64-encoded PNGs (example structure only — replace with real file contents):

```bash
IMG_B64=$(base64 -w0 your.png)
curl -s -X POST http://127.0.0.1:8080/layout-detection \
  -H "Content-Type: application/json" \
  -d "{\"images\":[\"$IMG_B64\"],\"layoutShapeMode\":\"auto\"}"
```

### ARM64 / Apple Silicon Docker

Building this **GPU Dockerfile** on **arm64** often fails or runs without NVIDIA GPU support because the **base image is aimed at amd64 + CUDA**. For ARM64, prefer the **native Linux ARM64** section above, or build a **CPU-only** image yourself (e.g. `FROM python:3.12-slim`) and install `torch`/`torchvision` for `aarch64` manually.

---

## Troubleshooting

| Issue | What to try |
|-------|----------------|
| `ImportError` / `torchvision` | `pip install torchvision` (matching your `torch` build). |
| `The serving plugin is not available` | `pip install filetype` and ensure `serving` extras are installed: `pip install -e ".[serving]"`. |
| Slow first start | Models download from Hugging Face; use `HF_HOME` or mount `~/.cache/huggingface` in Docker. |
| Meta tensor / `to()` errors | Use a recent `transformers` + ensure layout weights load with `device_map` (current `HFLayoutDetector` code paths). |

---

## Related files

| File | Role |
|------|------|
| `hf_server.py` | Pooling HTTP server (`/layout-parsing`, `/layout-detection`, …) |
| `deploy/docker/paddleocr-vl-hf/Dockerfile` | GPU Docker image |
| `paddlex/configs/pipelines/PaddleOCR-VL-HF.yaml` | Full HF layout + HF VLM |
| `paddlex/configs/pipelines/PaddleOCR-VL-HF-vllm.yaml` | HF layout + remote vLLM for VLM |
