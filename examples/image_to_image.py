import os
import requests
from io import BytesIO
from transformers import logging
from PIL import Image

from diffusion.constants import MODELS_PATH
from diffusion.pipelines.diffusion_img2img_pipeline import (
    DiffusionImg2ImgPipeline,
)

logging.set_verbosity_error()

url = "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/assets/stable-samples/img2img/sketch-mountains-input.jpg"

response = requests.get(url)
init_image = Image.open(BytesIO(response.content)).convert("RGB")
init_image.thumbnail((768, 768))

model_path = os.path.join(MODELS_PATH, "sdv1/pastelmix.safetensors")
pipeline = DiffusionImg2ImgPipeline(model_path)

image = pipeline(
    prompt="masterpiece, best quality, ultra-detailed, illustration, fantasy landscape",  # noqa: E501
    negative_prompt="lowres, bad anatomy, bad hands, text, missing finger, extra digits, fewer digits, blurry, mutated hands and fingers, poorly drawn face, mutation, deformed face, ugly, bad proportions, extra limbs, extra face, double head, extra head, extra feet, monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, jpeg artifacts",  # noqa: E501
    init_image=init_image,
    seed=42 + 31337,
    layer_skip=2,
)[0]

image.show()
