"""
RunPod Serverless Handler for CogVideoX-2B
Generates 6-second educational intro videos
Optimized for RTX 4090 (24GB VRAM)
"""
import os
import torch
import runpod
import base64
import uuid
import time
from diffusers import CogVideoXPipeline
from diffusers.utils import export_to_video

# Configuration
MODEL_ID = "THUDM/CogVideoX-2b"
DEFAULT_NUM_FRAMES = 49  # ~6 seconds at 8fps
DEFAULT_FPS = 8
DEFAULT_GUIDANCE_SCALE = 6.0
DEFAULT_NUM_INFERENCE_STEPS = 50
HF_CACHE_DIR = "/workspace/hf_cache"

# Intro scene prompt templates
INTRO_TEMPLATES = {
    "biology": "A stunning cinematic visualization of {topic}, vibrant colors, smooth camera movements, professional lighting, educational animation style",
    "chemistry": "An elegant molecular visualization showing {topic}, glowing particles, neon accents, smooth transitions, chemistry education style",
    "physics": "A breathtaking cosmic visualization of {topic}, dynamic motion, space aesthetics, stunning visual effects, physics education style",
    "mathematics": "A beautiful geometric animation showing {topic}, precise movements, elegant transitions, abstract patterns, mathematics education style",
    "default": "A high-quality educational animation showing {topic}, smooth transitions, clear visuals, professional quality, engaging educational style"
}

# Global pipeline (loaded once, reused across requests)
pipe = None


def get_intro_prompt(topic: str, subject: str = "default") -> str:
    """Generate optimized prompt for educational intro."""
    template = INTRO_TEMPLATES.get(subject.lower(), INTRO_TEMPLATES["default"])
    return template.format(topic=topic)


def load_model():
    """Load CogVideoX-2b model with optimizations for RTX 4090."""
    global pipe

    if pipe is not None:
        return pipe

    print("Loading CogVideoX-2b model...")
    start_time = time.time()

    # Set cache directory
    os.environ['HF_HOME'] = HF_CACHE_DIR

    pipe = CogVideoXPipeline.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float16,
        cache_dir=HF_CACHE_DIR
    )

    # Use CPU offload for memory efficiency on 24GB GPU
    pipe.enable_model_cpu_offload()

    # Enable VAE optimizations to prevent OOM during decoding
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    load_time = time.time() - start_time
    print(f"Model loaded in {load_time:.1f}s")

    return pipe


def generate_video(job):
    """Main handler function for RunPod serverless."""
    try:
        job_input = job.get("input", {})
        job_id = job.get("id", str(uuid.uuid4()))

        # Extract parameters
        prompt = job_input.get("prompt", "")
        topic = job_input.get("topic", "")
        subject = job_input.get("subject", "default")

        # Build prompt from topic if no direct prompt
        if topic and not prompt:
            prompt = get_intro_prompt(topic, subject)

        if not prompt:
            return {"error": "No prompt or topic provided", "status": "failed"}

        # Video parameters
        num_frames = job_input.get("num_frames", DEFAULT_NUM_FRAMES)
        guidance_scale = job_input.get("guidance_scale", DEFAULT_GUIDANCE_SCALE)
        num_steps = job_input.get("num_inference_steps", DEFAULT_NUM_INFERENCE_STEPS)
        seed = job_input.get("seed", 42)

        print(f"[{job_id}] Generating intro video...")
        print(f"[{job_id}] Prompt: {prompt[:100]}...")
        print(f"[{job_id}] Frames: {num_frames}, Steps: {num_steps}")

        # Load model
        pipeline = load_model()

        # Generate video
        start_time = time.time()

        video_frames = pipeline(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
        ).frames[0]

        generation_time = time.time() - start_time
        print(f"[{job_id}] Video generated in {generation_time:.1f}s")

        # Save to temp file
        temp_path = f"/tmp/{job_id}_intro.mp4"
        export_to_video(video_frames, temp_path, fps=DEFAULT_FPS)

        # Read video as base64 for return
        with open(temp_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")

        # Get file size
        file_size = os.path.getsize(temp_path)

        # Clean up
        os.remove(temp_path)

        duration_seconds = num_frames / DEFAULT_FPS

        return {
            "job_id": job_id,
            "status": "completed",
            "video_base64": video_base64,
            "prompt": prompt,
            "num_frames": num_frames,
            "fps": DEFAULT_FPS,
            "duration_seconds": duration_seconds,
            "generation_time_seconds": round(generation_time, 1),
            "file_size_bytes": file_size
        }

    except Exception as e:
        print(f"Error generating video: {e}")
        import traceback
        traceback.print_exc()
        return {
            "job_id": job.get("id", "unknown"),
            "status": "failed",
            "error": str(e)
        }


# RunPod serverless handler
runpod.serverless.start({"handler": generate_video})

