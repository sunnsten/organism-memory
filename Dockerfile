FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/workspace/cache/hf \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-venv python-is-python3 \
    git ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

RUN python3 -m pip install --upgrade pip wheel setuptools

RUN python3 -m pip install --upgrade --index-url https://download.pytorch.org/whl/cu128 \
    torch torchvision torchaudio

COPY requirements.txt requirements-dev.txt requirements-analytics.txt ./

RUN python3 -m pip install -r requirements.txt

RUN python3 -m pip install -r requirements-analytics.txt

RUN python3 -m pip install -r requirements-dev.txt

COPY . /workspace

RUN mkdir -p /workspace/cache/hf

EXPOSE 8000

CMD ["uvicorn", "organism.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
