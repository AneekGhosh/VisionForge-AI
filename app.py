
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk
import torch
import uuid
import os
import shutil
import threading
import sys

from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    DPMSolverMultistepScheduler
)

# =========================
# CONFIG
# =========================

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH = "./realisticVisionV60B1_v51HyperVAE.safetensors"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

WIDTH = 512
HEIGHT =512

INFERENCE_STEPS = 20
GUIDANCE_SCALE = 7.5

prompt_history = []


STYLE_PRESETS = {
    "Realistic": "ultra realistic, highly detailed, professional photography",
    "Anime": "anime style, detailed anime illustration, studio quality",
    "Fantasy": "fantasy art, magical atmosphere, epic fantasy scene",
    "Cyberpunk": "cyberpunk city, neon lights, futuristic technology",
    "Luxury": "luxury lifestyle, cinematic lighting, premium quality"
}



# =========================
# SAFETY CHECKER DISABLE
# =========================

def dummy_checker(images, **kwargs):
    return images, [False] * len(images)

# =========================
# LOAD MODEL
# =========================

print("Loading AI model...")

try:
    pipe = StableDiffusionPipeline.from_single_file(
        MODEL_PATH,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker=None
    )

    # Faster scheduler
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(
        pipe.scheduler.config
    )

    pipe = pipe.to(DEVICE)

    pipe.safety_checker = dummy_checker

    # Optimizations
    pipe.enable_attention_slicing()
    pipe.enable_vae_slicing()

    if DEVICE == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

        try:
            pipe.enable_xformers_memory_efficient_attention()
        except:
            pass

    print(f"Model loaded successfully on {DEVICE.upper()}!")

except Exception as e:
    print(f"Error loading model: {e}")
    messagebox.showerror(
        "Model Load Error",
        f"Failed to load model:\n{e}"
    )
    sys.exit(1)

# =========================
# GLOBALS
# =========================

img2img_pipe = None
uploaded_image_path = None
last_output_path = None

# =========================
# GUI
# =========================

app = tk.Tk()
app.title("VisionForge AI")
app.geometry("750x800")
app.configure(bg="#2e2e2e")

style = ttk.Style(app)
style.theme_use("clam")
style.configure(
    "Vision.Horizontal.TProgressbar",
    troughcolor="#3a3a3a",
    background="#00D4FF",
    bordercolor="#3a3a3a",
    lightcolor="#00D4FF",
    darkcolor="#00D4FF"
)

style.configure(
    "TLabel",
    background="#2e2e2e",
    foreground="white",
    font=("Arial", 10)
)

style.configure(
    "TButton",
    font=("Arial", 10, "bold")
)

prompt_var = tk.StringVar()
style_var = tk.StringVar(value="Realistic")
resolution_var = tk.StringVar(value="512x512")
steps_var = tk.IntVar(value=20)
guidance_var = tk.DoubleVar(value=7.5)
progress_var = tk.StringVar()
progress_percent_var = tk.StringVar(value="0%")
output_path_var = tk.StringVar(value="No image generated yet.")

negative_prompt_var = tk.StringVar(
    value="blurry, ugly, distorted, low quality"
)




# =========================
# HELPERS
# =========================

def browse_image():
    global uploaded_image_path

    file_path = filedialog.askopenfilename(
        title="Select Base Image",
        filetypes=[
            ("Images", "*.png *.jpg *.jpeg *.webp")
        ]
    )

    if file_path:
        uploaded_image_path = file_path
        messagebox.showinfo(
            "Image Selected",
            os.path.basename(file_path)
        )

def enable_buttons():
    generate_btn.config(state=tk.NORMAL)
    browse_btn.config(state=tk.NORMAL)
    download_btn.config(state=tk.NORMAL)

def disable_buttons():
    generate_btn.config(state=tk.DISABLED)
    browse_btn.config(state=tk.DISABLED)
    download_btn.config(state=tk.DISABLED)

# =========================
# IMAGE GENERATION
# =========================

def progress_callback(step, timestep, latents):

    percent = int(((step + 1) / INFERENCE_STEPS) * 100)

    progress_bar["value"] = percent
    progress_percent_var.set(f"{percent}%")

    app.update_idletasks()


def generate_worker():

    global img2img_pipe
    global last_output_path

    try:

        prompt = prompt_var.get().strip()

        if prompt:
            prompt_history.insert(0, prompt)

            if len(prompt_history) > 10:
                prompt_history.pop()

        history_list.delete(0, tk.END)

        for item in prompt_history:

            history_list.insert(tk.END, item)


        selected_style = STYLE_PRESETS.get(
            style_var.get(),
            ""
        )

        prompt = f"{prompt}, {selected_style}"

        if not prompt:
            raise Exception("Prompt cannot be empty.")

        # =========================
        # IMG2IMG
        # =========================

        if uploaded_image_path:

            if img2img_pipe is None:

                print("Loading img2img pipeline...")

                img2img_pipe = StableDiffusionImg2ImgPipeline.from_single_file(
                    MODEL_PATH,
                    torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
                    safety_checker=None
                ).to(DEVICE)

                img2img_pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                    img2img_pipe.scheduler.config
                )

                img2img_pipe.safety_checker = dummy_checker

                img2img_pipe.enable_attention_slicing()

            init_img = (
                Image.open(uploaded_image_path)
                .convert("RGB")
                .resize((WIDTH, HEIGHT))
            )

            result = img2img_pipe(
                prompt=prompt,
                negative_prompt=negative_prompt_var.get(),
                image=init_img,
                strength=0.75,
                guidance_scale=GUIDANCE_SCALE,
                num_inference_steps=INFERENCE_STEPS
            )

        # =========================
        # TXT2IMG
        # =========================

        else:
            selected_resolution = resolution_var.get()

            WIDTH, HEIGHT = map(
                int,
                selected_resolution.split("x")
            )

            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt_var.get(),
                height=HEIGHT,
                width=WIDTH,
                num_inference_steps=INFERENCE_STEPS,
                guidance_scale=GUIDANCE_SCALE,
                callback=progress_callback,
                callback_steps=1




            )

        result_img = result.images[0]





        # =========================
        # SAVE
        # =========================

        filename = f"{uuid.uuid4().hex}.png"

        full_path = os.path.abspath(
            os.path.join(OUTPUT_DIR, filename)
        )

        result_img.save(full_path)

        last_output_path = full_path

        # =========================
        # DISPLAY
        # =========================

        preview = result_img.resize((300, 300))

        preview_tk = ImageTk.PhotoImage(preview)

        output_panel.config(
            image=preview_tk,
            text=""
        )

        output_panel.image = preview_tk

        progress_var.set("Generation Complete ✅")
        output_path_var.set(full_path)

        history_list.delete(0, tk.END)

        for item in prompt_history:
            history_list.insert(tk.END, item)


    except Exception as e:

        messagebox.showerror(
            "Generation Error",
            str(e)
        )

        print(e)

    finally:

        enable_buttons()

    progress_bar["value"] = 100
    progress_percent_var.set("100%")



# =========================
# GENERATE BUTTON
# =========================

def generate_image():

    if not prompt_var.get().strip():
        messagebox.showerror(
            "Missing Prompt",
            "Please enter a prompt."
        )
        return

    progress_var.set("Generating image...")

    disable_buttons()

    progress_bar["value"] = 0
    progress_percent_var.set("0%")

    threading.Thread(
        target=generate_worker,
        daemon=True
    ).start()

# =========================
# DOWNLOAD
# =========================

def download_image():

    global last_output_path

    if not last_output_path:
        messagebox.showerror(
            "No Image",
            "Generate an image first."
        )
        return

    save_path = filedialog.asksaveasfilename(
        defaultextension=".png",
        filetypes=[
            ("PNG", "*.png")
        ]
    )

    if save_path:

        shutil.copy(last_output_path, save_path)

        messagebox.showinfo(
            "Saved",
            f"Image saved to:\n{save_path}"
        )

# =========================
# UI LAYOUT
# =========================
title_label = tk.Label(
    app,
    text="VisionForge AI",
    font=("Segoe UI", 24, "bold"),
    fg="#00D4FF",
    bg="#2e2e2e"
)
title_label.pack(pady=(15, 0))

subtitle_label = tk.Label(
    app,
    text="Developed by Aneek Ghosh",
    font=("Segoe UI", 10),
    fg="white",
    bg="#2e2e2e"
)
subtitle_label.pack(pady=(0, 10))


ttk.Label(
    app,
    text="Enter Prompt:",
    font=("Arial", 12)
).pack(pady=(10, 5))

ttk.Entry(
    app,
    textvariable=prompt_var,
    width=90,
    font=("Arial", 11)
).pack(pady=5)

ttk.Label(
    app,
    text="Negative Prompt:",
    font=("Arial", 12)
).pack(pady=(10, 5))

ttk.Entry(
    app,
    textvariable=negative_prompt_var,
    width=90,
    font=("Arial", 11)
).pack(pady=5)

# =========================
# STYLE PRESET
# =========================

ttk.Label(
    app,
    text="Style Preset:",
    font=("Arial", 12)
).pack(pady=(10, 5))

style_dropdown = ttk.Combobox(
    app,
    textvariable=style_var,
    values=list(STYLE_PRESETS.keys()),
    state="readonly",
    width=35
)

style_dropdown.pack(pady=5)

ttk.Label(
    app,
    text="Resolution:",
    font=("Arial", 12)
).pack(pady=(10, 5))

resolution_dropdown = ttk.Combobox(
    app,
    textvariable=resolution_var,
    values=[
        "512x512",
        "768x768",
        "1024x768",
        "1024x1024"
    ],
    state="readonly",
    width=20
)

resolution_dropdown.pack(pady=5)


browse_btn = tk.Button(
    app,
    text="📁 Upload Base Image",
    command=browse_image,
    bg="#3498db",
    fg="white",
    font=("Segoe UI", 10, "bold"),
    width=20,
    height=1,
    relief="flat"
)

browse_btn.pack(pady=5)

generate_btn = tk.Button(
    app,
    text="✨Generate with VisionForge",
    command=generate_image,
    bg="#2ecc71",
    fg="white",
    font=("Segoe UI", 11, "bold"),
    width=25,
    height=1,
    relief="flat"
)

generate_btn.pack(pady=5)

progress_bar = ttk.Progressbar(
    app,
    orient="horizontal",
    length=300,
    mode="determinate",
    style="Vision.Horizontal.TProgressbar"
)

progress_bar.pack(pady=(8,2))

#ttk.Label(
#    app,
#    textvariable=progress_percent_var
#).pack()

ttk.Label(
    app,
    textvariable=progress_var,
    foreground="cyan"
).pack(pady=5)

ttk.Label(
    app,
    textvariable=output_path_var,
    wraplength=650
).pack(pady=5)

output_panel = ttk.Label(
    app,
    text="Generated Image Appears Here",
    background="#444444",
    foreground="white",
    anchor="center"
)

output_panel.pack(
    pady=20,
    padx=20,
    fill="both",
    expand=True
)

# =========================
# PROMPT HISTORY
# =========================

history_frame = tk.LabelFrame(
    app,
    text="Prompt History",
    bg="#2e2e2e",
    fg="white"
)

history_frame.pack(
    pady=10,
    padx=20,
    fill="x"
)

history_list = tk.Listbox(
    history_frame,
    width=45,
    height=5
)

history_list.pack(
    padx=5,
    pady=5,
    fill="x"
)

def load_history_prompt(event):

    selection = history_list.curselection()

    if selection:
        prompt_var.set(
            history_list.get(selection[0])
        )

history_list.bind(
    "<Double-Button-1>",
    load_history_prompt
)




download_btn = tk.Button(
    app,
    text="💾 Save Generated Image",
    command=download_image,
    bg="#9b59b6",
    fg="white",
    activebackground="#8e44ad",
    activeforeground="white",
    font=("Segoe UI", 10, "bold"),
    width=20,
    height=1,
    relief="flat"
)

download_btn.pack(pady=15)

# =========================
# MAIN LOOP
# =========================

footer = tk.Label(
    app,
    text="VisionForge AI © 2026 | Developed by Aneek Ghosh",
    font=("Arial", 8),
    fg="gray",
    bg="#2e2e2e"
)
footer.pack(side="bottom", pady=5)

app.mainloop()

