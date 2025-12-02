# CogVideoX-2B RunPod Serverless Deployment
# Using RunPod's official base image (already on Docker Hub = fast push)

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install only the additional Python packages we need
# (PyTorch, CUDA, ffmpeg are already in the base image)
RUN pip install --no-cache-dir \
    diffusers \
    transformers \
    accelerate \
    sentencepiece \
    imageio \
    imageio-ffmpeg

# Copy handler code
COPY handler.py /app/handler.py

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV HF_HOME=/runpod-volume
ENV TRANSFORMERS_CACHE=/runpod-volume

# RunPod handler entrypoint
CMD ["python", "-u", "handler.py"]

