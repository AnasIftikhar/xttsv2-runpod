FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    portaudio19-dev \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip
RUN pip3 install --no-cache-dir --upgrade pip

# Install PyTorch with CUDA 11.8
RUN pip3 install --no-cache-dir \
    torch==2.1.1+cu118 \
    torchaudio==2.1.1+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# Install RunPod
RUN pip3 install --no-cache-dir runpod

# Install XTTS API Server
RUN pip3 install --no-cache-dir xtts-api-server

# Pre-download the XTTS model (this prevents first-run timeout)
RUN python3 -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False)" || true

# Create necessary directories
RUN mkdir -p /app/speakers /app/output /app/models

# Copy handler
COPY rp_handler.py /app/rp_handler.py

EXPOSE 8020

CMD ["python3", "-u", "rp_handler.py"]
