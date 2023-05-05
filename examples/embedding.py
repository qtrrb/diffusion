import os
from transformers import logging

from diffusion.utils.constants import MODELS_PATH, EMBEDDINGS_PATH
from diffusion.pipelines.diffusion_txt2img_pipeline import (
    DiffusionTxt2ImgPipeline,
)
from diffusion.samplers.ksampler import KSampler

from diffusion.modules.textual_inversion import (
    Embedding,
)

logging.set_verbosity_error()

model_path = os.path.join(MODELS_PATH, "sdv1/pastelmix.safetensors")
embedding_path = os.path.join(EMBEDDINGS_PATH, "EasyNegative.safetensors")
pipeline = DiffusionTxt2ImgPipeline(model_path)

sampler = KSampler(pipeline.model, "sample_dpmpp_2m")

embedding = Embedding(embedding_path)

prompt = "masterpiece, best quality, ultra-detailed, illustration, portrait, 1girl"
negative_prompt = "EasyNegative"

image = pipeline(
    prompt=prompt,
    negative_prompt=negative_prompt,
    seed=42 + 31337,
    sampler=sampler,
    steps=25,
    H=640,
    W=448,
    layer_skip=2,
    scale=10,
    embedding=embedding,
)

image.show()
