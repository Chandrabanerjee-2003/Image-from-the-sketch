import io
import base64
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from groq import Groq

# ==========================================
# 1. YOUR SETTINGS
# ==========================================
# Replace this with your actual Groq API Key
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE" 

app = FastAPI()

# Allow your frontend to talk to this backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImagePayload(BaseModel):
    image_base64: str

# Global variables for models
sd_pipe = None
groq_client = Groq(api_key=GROQ_API_KEY)

# ==========================================
# 2. MODEL LOADING (WITH AUTO-GPU/CPU CHECK)
# ==========================================
def load_sd_pipeline():
    print("--- STEP 1: Checking Hardware ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use half-precision (float16) only if on GPU to save memory
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    print(f"--- STEP 2: Loading AI Model to {device.upper()} ---")
    print("This takes 1-2 minutes. Please wait...")
    
    try:
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        # Load UNet
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

        # Load Pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, 
            unet=unet, 
            torch_dtype=dtype, 
            variant="fp16" if device == "cuda" else None
        ).to(device)

        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        print("--- SUCCESS: Backend is ready! ---")
        return pipe
    except Exception as e:
        print(f"--- ERROR DURING LOADING: {e} ---")
        return None

@app.on_event("startup")
async def startup_event():
    global sd_pipe
    sd_pipe = load_sd_pipeline()

# ==========================================
# 3. THE PROCESSING LOGIC
# ==========================================
@app.post("/process")
async def process_drawing(payload: ImagePayload):
    if sd_pipe is None:
        raise HTTPException(status_code=500, detail="AI Model is still loading or failed to load. Check terminal.")

    try:
        # Decode the image from the canvas
        image_data = base64.b64decode(payload.image_base64)
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # Get description from Groq Vision
        buffered = io.BytesIO()
        input_image.save(buffered, format="JPEG")
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")
        
        print("Asking Groq to describe your drawing...")
        chat_completion = groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this sketch artistically in 20 words for a refined digital painting. Mention colors."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ],
            }],
            model="llama-3.2-11b-vision-preview",
        )
        prompt = chat_completion.choices[0].message.content
        print(f"Prompt Generated: {prompt}")

        # Generate the new image
        # num_inference_steps=4 for Lightning model
        print("Generating AI image...")
        generated_image = sd_pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]

        # Convert back to base64 for the frontend
        out_buffered = io.BytesIO()
        generated_image.save(out_buffered, format="JPEG")
        gen_image_b64 = base64.b64encode(out_buffered.getvalue()).decode("utf-8")

        return {"generated_image_base64": gen_image_b64}

    except Exception as e:
        print(f"Processing Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Note: filename MUST be backend.py for this to work
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
