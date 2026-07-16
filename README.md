# 🎨 Sketch-to-Image AI Generator

An AI-powered web application that transforms hand-drawn sketches into realistic images using **Groq's Llama 3.2 Vision** and **Stable Diffusion XL**. Users can draw directly on an HTML5 Canvas, and the application automatically interprets the sketch, generates a detailed textual description, and produces a photorealistic image.

---

# 🚀 Features

* 🖌️ Interactive HTML5 Canvas drawing interface
* 🎨 Multiple brush colors and adjustable brush size
* 🧽 Pen, Eraser, and Clear Canvas tools
* 📱 Mouse and touch device support
* 🧠 Automatic sketch understanding using **Groq Llama 3.2 Vision**
* 🖼️ High-quality image generation using **Stable Diffusion XL**
* ⚡ Fast REST API backend built with **FastAPI**
* 🔄 Real-time frontend-backend communication using Fetch API

---

# 🏗️ System Architecture

```text
                 User
                   │
                   ▼
        HTML5 Canvas (Frontend)
                   │
         Draw Sketch (Pixels)
                   │
                   ▼
      Convert Canvas → JPEG → Base64
                   │
                   ▼
        HTTP POST Request (JSON)
                   │
                   ▼
              FastAPI Backend
                   │
        Decode Base64 → PIL Image
                   │
                   ▼
      Groq Llama 3.2 Vision API
       (Sketch → Text Prompt)
                   │
                   ▼
        Stable Diffusion XL
      (Prompt → Realistic Image)
                   │
                   ▼
     PIL Image → JPEG → Base64
                   │
                   ▼
        Frontend Displays Result
```

---

# 🧠 How It Works

### Step 1 – User Draws

The user sketches directly on an HTML5 Canvas using drawing tools such as:

* Pen
* Eraser
* Color Palette
* Brush Size Slider

Canvas stores the drawing as pixels.

---

### Step 2 – Canvas Serialization

JavaScript converts the canvas into a JPEG image using:

```javascript
canvas.toDataURL("image/jpeg")
```

The JPEG is then Base64 encoded and sent to the backend as a JSON payload.

Example:

```json
{
  "image_base64": "..."
}
```

---

### Step 3 – FastAPI Backend

The backend:

* Receives the POST request
* Decodes the Base64 string
* Converts it into a PIL Image
* Sends the image to the Groq Vision model

---

### Step 4 – Sketch Understanding

The sketch is analyzed using **Groq Llama 3.2 Vision**, which:

* Understands the objects in the sketch
* Generates a rich textual description

Example:

```
Input Sketch:
(Simple drawing of a cat)

↓

Generated Prompt:

"A fluffy orange cat sitting on green grass under bright sunlight."
```

---

### Step 5 – Image Generation

The generated prompt is passed to **Stable Diffusion XL**.

Stable Diffusion:

* Converts the prompt into text embeddings
* Starts with random noise
* Iteratively removes noise using a U-Net
* Uses cross-attention to align the image with the prompt
* Produces a realistic image

---

### Step 6 – Display Result

The generated image is:

* Converted into Base64
* Sent back to the frontend
* Rendered dynamically without refreshing the page

---

# 🛠️ Tech Stack

## Frontend

* HTML5
* CSS3
* JavaScript (ES6)
* HTML5 Canvas API
* Fetch API

## Backend

* Python
* FastAPI
* Uvicorn
* Pillow (PIL)
* Base64
* BytesIO

## AI Models

* Groq Llama 3.2 Vision
* Stable Diffusion XL

---

# 📂 Project Structure

```text
.
├── index.html          # Frontend UI
├── server.py           # FastAPI backend
├── static/
├── assets/
├── requirements.txt
├── README.md
└── .env
```

---

# ⚙️ Installation

## 1. Clone the Repository

```bash
git clone https://github.com/your-username/sketch-to-image-ai.git

cd sketch-to-image-ai
```

---

## 2. Create Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Configure Environment Variables

Create a `.env` file:

```env
GROQ_API_KEY=your_api_key
```

---

## 5. Run Backend

```bash
python server.py
```

or

```bash
uvicorn server:app --reload
```

---

## 6. Run Frontend

Open `index.html`

or

serve it using a local server.

---

# 📡 API

## POST `/process`

Processes a hand-drawn sketch.

### Request

```json
{
  "image_base64": "..."
}
```

### Response

```json
{
  "description": "A fluffy orange cat sitting on green grass.",
  "generated_image_base64": "..."
}
```

---

# 📸 Workflow

```
User Sketch
      │
      ▼
Canvas
      │
      ▼
JPEG
      │
      ▼
Base64
      │
      ▼
FastAPI
      │
      ▼
Groq Vision
      │
      ▼
Prompt
      │
      ▼
Stable Diffusion XL
      │
      ▼
Generated Image
      │
      ▼
Frontend Display
```

---

# 💡 Key Concepts Used

* HTML5 Canvas API
* Event-Driven JavaScript
* REST APIs
* FastAPI
* Base64 Encoding & Decoding
* PIL Image Processing
* Vision-Language Models (VLMs)
* Image Embeddings
* Transformer Architecture
* Self-Attention
* Stable Diffusion
* Cross-Attention
* U-Net
* Latent Diffusion

---

# 🔮 Future Improvements

* Upload existing images for editing
* Undo/Redo functionality
* Download generated images
* User authentication
* Drawing history
* Image style selection
* Prompt editing before generation
* Cloud deployment with Docker

