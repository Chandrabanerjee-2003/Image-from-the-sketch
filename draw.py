import io
import base64
import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from groq import Groq
import uvicorn

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- CONFIGURATION ---
GROQ_API_KEY = 'YOUR_GROQ_API_KEY_HERE' # <--- PUT YOUR KEY HERE
device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Global models
sd_pipe = None
groq_client = Groq(api_key=GROQ_API_KEY)

class ImagePayload(BaseModel):
    image_base64: str

def load_sd_pipeline():
    print(f"Loading model to {device}...")
    base = "stabilityai/stable-diffusion-xl-base-1.0"
    repo = "ByteDance/SDXL-Lightning"
    ckpt = "sdxl_lightning_4step_unet.safetensors"

    try:
        # Load UNet
        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, torch_dtype)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

        # Load Pipeline
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=torch_dtype, variant="fp16"
        ).to(device)

        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        print("Model loaded successfully!")
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

@app.on_event("startup")
async def startup_event():
    global sd_pipe
    sd_pipe = load_sd_pipeline()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.post("/process")
async def process_drawing(payload: ImagePayload):
    if sd_pipe is None:
        raise HTTPException(status_code=500, detail="Model not loaded on server")

    try:
        # 1. Decode Canvas Image
        img_str = payload.image_base64.split(",")[-1]
        image_data = base64.b64decode(img_str)
        input_image = Image.open(io.BytesIO(image_data)).convert("RGB")

        # 2. Get Description from Groq
        buffered = io.BytesIO()
        input_image.save(buffered, format="JPEG")
        b64_image = base64.b64encode(buffered.getvalue()).decode("utf-8")

        chat_completion = groq_client.chat.completions.create(
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this sketch artistically in 20 words for a high-quality digital painting. Mention colors and lighting."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}}
                ],
            }],
            model="llama-3.2-11b-vision-preview",
        )
        prompt = chat_completion.choices[0].message.content
        print(f"Generated Prompt: {prompt}")

        # 3. Generate Image
        # num_inference_steps=4 matches the Lightning 4-step model
        output = sd_pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]

        # 4. Return as Base64
        out_buffered = io.BytesIO()
        output.save(out_buffered, format="JPEG")
        gen_image_b64 = base64.b64encode(out_buffered.getvalue()).decode("utf-8")

        return {"generated_image_base64": gen_image_b64}

    except Exception as e:
        print(f"Process Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("backend:app", host="0.0.0.0", port=8000, reload=True)
