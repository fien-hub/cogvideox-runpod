"""
RunPod Serverless Handler for CogVideoX
"""
import runpod
import torch
import base64
import tempfile
import os

# Global model - loaded once on cold start
pipe = None

def load_model():
    """Load CogVideoX model (only once per worker)"""
    global pipe
    if pipe is None:
        print("Loading CogVideoX model...")
        from diffusers import CogVideoXPipeline
        
        pipe = CogVideoXPipeline.from_pretrained(
            "THUDM/CogVideoX-2b",
            torch_dtype=torch.float16
        )
        pipe.to("cuda")
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        print("Model loaded successfully!")
    return pipe

def handler(job):
    """Handle inference request"""
    job_input = job["input"]
    
    # Get parameters
    prompt = job_input.get("prompt", "A beautiful sunset over the ocean")
    num_frames = job_input.get("num_frames", 49)
    guidance_scale = job_input.get("guidance_scale", 6.0)
    num_inference_steps = job_input.get("num_inference_steps", 50)
    fps = job_input.get("fps", 8)
    
    try:
        # Load model
        pipe = load_model()
        
        # Generate video
        print(f"Generating video for: {prompt}")
        video = pipe(
            prompt=prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        ).frames[0]
        
        # Save to temp file
        from diffusers.utils import export_to_video
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            temp_path = f.name
        
        export_to_video(video, temp_path, fps=fps)
        
        # Read and encode as base64
        with open(temp_path, "rb") as f:
            video_base64 = base64.b64encode(f.read()).decode("utf-8")
        
        # Cleanup
        os.remove(temp_path)
        
        return {
            "video_base64": video_base64,
            "prompt": prompt,
            "num_frames": num_frames,
            "fps": fps
        }
        
    except Exception as e:
        return {"error": str(e)}

# Start the serverless handler
runpod.serverless.start({"handler": handler})
