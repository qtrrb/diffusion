import os
import requests
from PIL import Image
from transformers import logging
from io import BytesIO

from diffusion.utils.constants import CONTROL_MODELS_PATH, MODELS_PATH
from diffusion.pipelines.diffusion_controlnet_pipeline import (
    DiffusionControlNetPipeline,
)

logging.set_verbosity_error()

model_path = os.path.join(MODELS_PATH, "sdv1/pastelmix.safetensors")
control_model_path = os.path.join(CONTROL_MODELS_PATH, "control_v11p_sd15_scribble.pth")
pipeline = DiffusionControlNetPipeline(
    "scribble",
    control_model_path,
    model_path,
)

url = "https://huggingface.co/datasets/YiYiXu/controlnet-testing/resolve/main/drawing_before.png"
response = requests.get(url)
control_image = Image.open(BytesIO(response.content)).convert("RGB")


prompt = "masterpiece, best quality, 1girl, witch, broom"
negative_prompt = "lowres, bad anatomy, bad hands, text, missing finger, extra digits, fewer digits, blurry, mutated hands and fingers, poorly drawn face, mutation, deformed face, ugly, bad proportions, extra limbs, extra face, double head, extra head, extra feet, monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, jpeg artifacts"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=42 + 31337,
    steps=30,
    init_image=control_image,
    layer_skip=2,
)[0]
control_image.show()
image.show()
