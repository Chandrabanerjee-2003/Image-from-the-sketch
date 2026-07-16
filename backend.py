import io
import os
import base64
import torch
import uvicorn
import logging
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from groq import Groq

# Setup Logging so you can see exactly what's happening
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DynamicBackend")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ImagePayload(BaseModel):
    image_base64: str

# --- CONFIGURATION (Change these via environment variables if needed) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "YOUR_ACTUAL_GROQ_KEY_HERE")
MODEL_REPO = "ByteDance/SDXL-Lightning"
MODEL_CKPT = "sdxl_lightning_4step_unet.safetensors"
BASE_MODEL = "stabilityai/stable-diffusion-xl-base-1.0"

# Global Variables
pipe = None
client = Groq(api_key=GROQ_API_KEY)
is_loading = True

def get_optimal_device():
    if torch.cuda.is_available(): return "cuda", torch.float16
    if torch.backends.mps.is_available(): return "mps", torch.float32 # For Mac M1/M2/M3
    return "cpu", torch.float32

def initialize_models():
    global pipe, is_loading
    device_name, dtype = get_optimal_device()
    logger.info(f"Initializing AI on device: {device_name}")
    
    try:
        # Load UNet
        unet = UNet2DConditionModel.from_config(BASE_MODEL, subfolder="unet").to(device_name, dtype)
        unet.load_state_dict(load_file(hf_hub_download(MODEL_REPO, MODEL_CKPT), device=device_name))
        
        # Load Pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            BASE_MODEL, 
            unet=unet, 
            torch_dtype=dtype, 
            use_safetensors=True
        ).to(device_name)
        
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        
        # Performance optimization
        if device_name == "cuda":
            pipe.enable_xformers_memory_efficient_attention()
            
        is_loading = False
        logger.info("✨ AI Models loaded and ready to generate!")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        is_loading = False

@app.on_event("startup")
async def startup_event():
    # Run initialization in a separate thread so the server starts IMMEDIATELY
    import threading
    threading.Thread(target=initialize_models).start()

@app.get("/health")
def health():
    return {"status": "online", "ai_ready": pipe is not None}

@app.post("/process")
async def process(payload: ImagePayload):
    if pipe is None:
        raise HTTPException(status_code=503, detail="AI model is still loading. Please wait 30 seconds.")

    try:
        # Clean up image string
        img_str = payload.image_base64.split(",")[-1]
        raw_img = Image.open(io.BytesIO(base64.b64decode(img_str))).convert("RGB")
        
        # Step 1: Vision Analysis (Dynamic prompt generation)
        logger.info("Analyzing drawing with Vision...")
        buf = io.BytesIO()
        raw_img.save(buf, format="JPEG")
        b64_vision = base64.b64encode(buf.getvalue()).decode("utf-8")

        try:
            chat = client.chat.completions.create(
                model="llama-3.2-11b-vision-preview",
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": "What is this? Describe it as a beautiful detailed digital painting in 15 words."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_vision}"}}
                ]}]
            )
            prompt = chat.choices[0].message.content
        except Exception as v_err:
            logger.warning(f"Vision failed, using default prompt: {v_err}")
            prompt = "A high-quality digital painting of the user's sketch, masterpiece, vibrant colors"

        logger.info(f"Generating with prompt: {prompt}")

        # Step 2: Image Generation
        # The num_inference_steps is set to 4 to match the SDXL-Lightning 4-step unet
        output = pipe(prompt=prompt, num_inference_steps=4, guidance_scale=0).images[0]

        # Step 3: Encode and Return
        out_buf = io.BytesIO()
        output.save(out_buf, format="JPEG")
        encoded_res = base64.b64encode(out_buf.getvalue()).decode("utf-8")

        return {"generated_image_base64": encoded_res}

    except Exception as e:
        logger.error(f"Process error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Dynamically find an open port or use 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)
