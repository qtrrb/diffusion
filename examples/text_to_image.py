import os
from transformers import logging

from diffusion.constants import MODELS_PATH
from diffusion.pipelines.diffusion_txt2img_pipeline import (
    DiffusionTxt2ImgPipeline,
)
from diffusion.ksampler import KSampler

logging.set_verbosity_error()

model_path = os.path.join(MODELS_PATH, "sdv1/pastelmix.safetensors")
pipeline = DiffusionTxt2ImgPipeline(model_path)
sampler = KSampler(pipeline.model, "sample_dpmpp_2m")

image = pipeline(
    prompt="masterpiece, best quality, ultra-detailed, illustration, portrait, 1girl",  # noqa: E501
    negative_prompt="lowres, bad anatomy, bad hands, text, missing finger, extra digits, fewer digits, blurry, mutated hands and fingers, poorly drawn face, mutation, deformed face, ugly, bad proportions, extra limbs, extra face, double head, extra head, extra feet, monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, jpeg artifacts",  # noqa: E501
    seed=42 + 31337,
    sampler=sampler,
    steps=20,
    H=640,
    W=448,
    layer_skip=2,
)[0]

image.show()
