# CogVideoX-2B RunPod Serverless Deployment
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install Python dependencies
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
