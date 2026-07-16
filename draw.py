import io, base64, torch, uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from groq import Groq

# ==========================================
# 1. SET YOUR API KEY HERE
# ==========================================
GROQ_API_KEY = "PASTE_YOUR_GROQ_KEY_HERE" 

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ImagePayload(BaseModel):
    image_base64: str

# Global Variables
pipe = None
client = Groq(api_key=GROQ_API_KEY)

def load_model():
    global pipe
    print("--- Loading AI Model (This may take a while) ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # Use float16 only if using CUDA (GPU) to save memory
    dtype = torch.float16 if device == "cuda" else torch.float32
    
    try:
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))
        
        pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=dtype, variant="fp16"
        ).to(device)
        
        pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config, timestep_spacing="trailing")
        print(f"--- SUCCESS: Model loaded on {device} ---")
    except Exception as e:
        print(f"--- ERROR LOADING MODEL: {e} ---")

@app.on_event("startup")
async def startup():
    load_model()

@app.post("/process")
async def process(payload: ImagePayload):
    if pipe is None:
        return {"error": "Model failed to load. Check your terminal/GPU memory."}
    
    try:
        # Decode Drawing
        img_data = base64.b64decode(payload.image_base64.split(",")[-1])
        input_img = Image.open(io.BytesIO(img_data)).convert("RGB")

        # Step 1: Vision Model (Groq)
        print("Asking Groq to describe drawing...")
        img_byte_arr = io.BytesIO()
        input_img.save(img_byte_arr, format='JPEG')
        b64_img = base64.b64encode(img_byte_arr.getvalue()).decode('utf-8')

        completion = client.chat.completions.create(
            model="llama-3.2-11b-vision-preview",
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this sketch in 15 words for a realistic digital art prompt."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}]
        )
        prompt = completion.choices[0].message.content
        print(f"Prompt Generated: {prompt}")

        # Step 2: Image Generation
        print("Generating Image...")
        # Use simple settings to avoid crash
        output = pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
        
        # Encode Result
        buf = io.BytesIO()
        output.save(buf, format="JPEG")
        return {"image": base64.b64encode(buf.getvalue()).decode('utf-8'), "prompt": prompt}

    except Exception as e:
        print(f"Processing Error: {e}")
        return {"error": str(e)}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
