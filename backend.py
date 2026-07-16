import io, base64, torch, uvicorn, threading
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PIL import Image
from diffusers import StableDiffusionXLPipeline, UNet2DConditionModel, EulerDiscreteScheduler
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from groq import Groq

# --- CONFIG ---
GROQ_API_KEY = "YOUR_GROQ_API_KEY_HERE"

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ImagePayload(BaseModel):
    image_base64: str

# Global State
sd_pipe = None
model_ready = False
status_msg = "Server is starting... please wait."
groq_client = Groq(api_key=GROQ_API_KEY)

# This function loads the heavy AI in the background
def load_ai_model():
    global sd_pipe, model_ready, status_msg
    try:
        print("--- [STARTING] Loading AI Model in background ---")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        
        base = "stabilityai/stable-diffusion-xl-base-1.0"
        repo = "ByteDance/SDXL-Lightning"
        ckpt = "sdxl_lightning_4step_unet.safetensors"

        unet = UNet2DConditionModel.from_config(base, subfolder="unet").to(device, dtype)
        unet.load_state_dict(load_file(hf_hub_download(repo, ckpt), device=device))

        sd_pipe = StableDiffusionXLPipeline.from_pretrained(
            base, unet=unet, torch_dtype=dtype, variant="fp16" if device == "cuda" else None
        ).to(device)

        sd_pipe.scheduler = EulerDiscreteScheduler.from_config(sd_pipe.scheduler.config, timestep_spacing="trailing")
        
        model_ready = True
        status_msg = "READY"
        print("--- [SUCCESS] AI Model Loaded and Ready! ---")
    except Exception as e:
        status_msg = f"ERROR: {str(e)}"
        print(f"--- [FAILED] {status_msg} ---")

# Start loading WITHOUT blocking the server
threading.Thread(target=load_ai_model).start()

@app.get("/")
def check_status():
    return {"status": status_msg, "ready": model_ready}

@app.post("/process")
async def process(payload: ImagePayload):
    if not model_ready:
        # This tells the frontend exactly why it's not working yet
        raise HTTPException(status_code=503, detail=status_msg)
    
    try:
        # 1. Vision
        img_data = base64.b64decode(payload.image_base64.split(",")[-1])
        input_img = Image.open(io.BytesIO(img_data)).convert("RGB")
        buf = io.BytesIO(); input_img.save(buf, format="JPEG")
        b64_img = base64.b64encode(buf.getvalue()).decode("utf-8")

        chat = groq_client.chat.completions.create(
            messages=[{"role": "user", "content": [
                {"type": "text", "text": "Describe this sketch for a digital painting in 15 words. Mention colors."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}}
            ]}],
            model="llama-3.2-11b-vision-preview",
        )
        prompt = chat.choices[0].message.content
        print(f"Prompt: {prompt}")

        # 2. Generation
        output = sd_pipe(prompt, num_inference_steps=4, guidance_scale=0).images[0]
        
        out_buf = io.BytesIO(); output.save(out_buf, format="JPEG")
        return {"generated_image_base64": base64.b64encode(out_buf.getvalue()).decode("utf-8")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
