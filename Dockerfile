FROM runpod/pytorch:2.2.0-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /app

# Install dependencies with specific versions for compatibility
RUN pip install --no-cache-dir \
    runpod \
    diffusers==0.30.3 \
    transformers==4.44.0 \
    accelerate==0.33.0 \
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
