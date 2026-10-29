import os
from flask import Flask, render_template, request, send_from_directory
from PIL import Image
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from diffusers.utils import load_image
from controlnet_aux import CannyDetector

# ------------------------------
# FLASK SETUP
# ------------------------------
app = Flask(__name__)
UPLOAD_FOLDER = "show"
UPLOADS = os.path.join(UPLOAD_FOLDER, "uploads")
OUTPUTS = os.path.join(UPLOAD_FOLDER, "outputs")
os.makedirs(UPLOADS, exist_ok=True)
os.makedirs(OUTPUTS, exist_ok=True)

# ------------------------------
# DEVICE & MODELS
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32
print(f" Using device: {device}")

# Load ControlNet (Canny)
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny",
    torch_dtype=dtype
)

# Load Stable Diffusion
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype
)

pipe.enable_attention_slicing()

if device == "cuda":
    pipe.to(device)
    pipe.enable_model_cpu_offload()  # Saves GPU memory
else:
    pipe.to("cpu")

print("✅ Optimized ControlNet + Stable Diffusion loaded!")

# ------------------------------
# THEMES
# ------------------------------
themes = {
    "futuristic": "futuristic modern environment, metallic lights, sleek design",
    "nature": "eco-friendly, greenery, natural light, peaceful vibe",
    "cyberpunk": "cyberpunk neon glow, city lights, purple and blue hues",
}

# ------------------------------
# CANNY DETECTOR
# ------------------------------
canny = CannyDetector()

# ------------------------------
# ROUTES
# ------------------------------
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/generate", methods=["POST"])
def generate():
    try:
        file = request.files["image"]
        theme = request.form["theme"]

        input_path = os.path.join(UPLOADS, file.filename)
        file.save(input_path)

        output_filename = f"output_{theme}.png"
        output_path = os.path.join(OUTPUTS, output_filename)

        # Resize smaller for speed
        input_image = load_image(input_path).resize((384, 384))
        control_image = canny(input_image)

        prompt = f"{themes[theme]}, realistic, detailed, high quality"

        print(f"🎨 Generating {theme} style...")

        result = pipe(
            prompt=prompt,
            image=control_image,
            num_inference_steps=8,     # reduced from 15
            guidance_scale=6,          # reduced from 7.5
        ).images[0]

        result.save(output_path)
        print(f"✅ Done: {output_path}")

        return render_template(
            "index.html",
            original=f"/show/uploads/{file.filename}",
            result=f"/show/outputs/{output_filename}",
            theme=theme,
        )

    except Exception as e:
        print(f"❌ Error: {e}")
        return f"Error occurred: {e}"

@app.route("/show/<path:filename>")
def serve_image(filename):
    return send_from_directory("show", filename)

# ------------------------------
# RUN
# ------------------------------
if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
