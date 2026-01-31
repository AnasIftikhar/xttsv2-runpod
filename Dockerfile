FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}

# CRITICAL: Auto-accept Coqui TTS Terms of Service
ENV COQUI_TOS_AGREED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    wget \
    ffmpeg \
    libsndfile1 \
    portaudio19-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Upgrade pip and install build tools
RUN pip3 install --no-cache-dir --upgrade pip setuptools wheel

# Install PyTorch with CUDA support
RUN pip3 install --no-cache-dir \
    torch==2.1.1 \
    torchaudio==2.1.1 \
    --index-url https://download.pytorch.org/whl/cu118

# Install TTS and its dependencies
RUN pip3 install --no-cache-dir \
    TTS \
    runpod \
    requests \
    numpy \
    scipy \
    librosa \
    soundfile \
    inflect \
    unidecode

# Pre-download the XTTS model (TOS will be auto-accepted via env variable)
RUN python3 -c "from TTS.api import TTS; print('Downloading XTTS v2 model...'); tts = TTS('tts_models/multilingual/multi-dataset/xtts_v2', gpu=False); print('Model cached successfully!')"

# Create necessary directories
RUN mkdir -p /app /tmp/tts_cache

# Set environment variable for TTS cache
ENV TTS_CACHE=/tmp/tts_cache

# Copy the handler file
COPY rp_handler.py /app/rp_handler.py

# Make handler executable
RUN chmod +x /app/rp_handler.py

# Expose port (for documentation purposes)
EXPOSE 8020

# Run the handler
CMD ["python3", "-u", "/app/rp_handler.py"]
