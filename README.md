# 🎨  ImageGeneration-from-the-sketch

**Canvas to Diffusion** is a web app that allows users to **sketch rough drawings on a canvas**, describes them using a **vision-language model (Groq + LLaMA 3 Vision)**, and then transforms the description into an **artistic image using Stable Diffusion XL**. The app combines deep learning, frontend interactivity, and backend orchestration to deliver a seamless sketch-to-art experience.

---

##  Features

-  Interactive sketch canvas with color and brush controls
-  Light and Dark mode toggle
-  Vision-based image description using **Groq's LLaMA-3 Vision model**
-  AI-generated art using **Stable Diffusion XL (via diffusers)**
-  Real-time image processing between frontend and FastAPI backend
-  Eraser, clear, and brush size tools for better control

---

##  Tech Stack

###  Backend
- **Python**
- **FastAPI** — Web API framework
- **diffusers** — Load Stable Diffusion XL models
- **torch** + **safetensors** — Efficient model loading and inference
- **Pillow (PIL)** — Image decoding and processing
- **Groq API** — Vision + Language captioning from sketches
- **huggingface_hub** — Load models and checkpoints

###  Frontend
- **HTML5 + Canvas** — Drawing interface
- **CSS3** — Aesthetic and responsive design (light pink theme + dark mode)
- **JavaScript (Vanilla)** — Canvas drawing, tool control, API interaction

---

##  Installation
1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/canvas-to-diffusion.git
cd canvas-to-diffusion

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/canvas-to-diffusion.git
cd canvas-to-diffusion
