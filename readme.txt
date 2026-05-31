VisionForge-AI
AI-powered text-to-image generator built with Python, Stable Diffusion, PyTorch, and CUDA acceleration.
Generate high-quality AI images locally using the Realistic Vision v6.0 model with both desktop GUI and future full-stack support.

Features
Text-to-Image generation (txt2img)
Image-to-Image generation (img2img)
Stable Diffusion integration
CUDA GPU acceleration support
Fast local AI inference
Tkinter desktop GUI
Custom model loading (.safetensors)
Negative prompt support
Output image saving & preview
Fully offline after setup

Tech Stack
Python 3.10
PyTorch
Diffusers
Stable Diffusion
Tkinter
CUDA
Pillow
Transformers

System Requirements
Recommended
NVIDIA GPU with CUDA support
8GB+ RAM
Python 3.10
Minimum
CPU supported (slower generation)
4GB RAM

Installation
1. Clone Repository
git clone https://github.com/AneekGhosh/VisionForge-AI.git
cd VisionForge-AI

2. Create Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
Linux/macOS
python3 -m venv .venv
source .venv/bin/activate

3. Install Dependencies
pip install -r requirements.txt

Model Setup
Download the Realistic Vision V6.0 model:
Model file:
realisticVisionV60B1_v51HyperVAE.safetensors
Place the model file inside the project root directory.

Run the Project
python app.py

Current Capabilities
512x512 AI image generation
Local GPU inference
Prompt-based generation
Custom negative prompts
Image preview & save

Future Roadmap
Full-stack MERN/Spring Boot integration
High-resolution image generation
User authentication
AI image gallery
Prompt history
Cloud deployment
REST API support
AI image marketplace/community features

Sample Prompts
Cyberpunk city at night, ultra realistic, cinematic lighting
Fantasy warrior girl, detailed face, 8k, sharp focus
Modern futuristic luxury villa, realistic architecture

Disclaimer
This project is for educational and research purposes only.
Users are responsible for generated content and usage.

Author
Developed by Aneek Ghosh
GitHub: https://github.com/AneekGhosh