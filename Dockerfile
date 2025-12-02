FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    runpod \
    diffusers>=0.30.0 \
    transformers \
    accelerate \
    sentencepiece \
    imageio[ffmpeg] \
    imageio \
    opencv-python-headless

# Copy handler
COPY handler.py /app/handler.py

# Set environment
ENV HF_HOME=/runpod-volume

# Run handler
CMD ["python", "-u", "/app/handler.py"]
