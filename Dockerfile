# CogVideoX-2B RunPod Serverless - Simple Base
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install PyTorch with CUDA and other packages
RUN pip3 install --no-cache-dir \
    torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118

RUN pip3 install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    imageio \
    imageio-ffmpeg \
    runpod

# Copy handler code
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume

# RunPod handler entrypoint
CMD ["python3", "-u", "handler.py"]
