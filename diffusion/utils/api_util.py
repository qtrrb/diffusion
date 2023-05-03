import io
import os
import requests
from PIL import Image
from ..api.schemas import TextArgs, ImageArgs
from .constants import MODELS_PATH, VAES_PATH, LORAS_PATH, EMBEDDINGS_PATH
from ..pipelines.diffusion_txt2img_pipeline import DiffusionTxt2ImgPipeline
from ..pipelines.diffusion_img2img_pipeline import DiffusionImg2ImgPipeline
from ..pipelines.diffusion_upscaling_pipeline import DiffusionUpscalingPipeline
from ..samplers.ddim import DDIMSampler
from ..samplers.dpm_solver import DPMSolverSampler
from ..samplers.plms import PLMSSampler
from ..samplers.ksampler import KSampler
from ..modules.lora import Lora
from ..modules.textual_inversion import Embedding


def convert_url_to_img(url: str) -> Image:
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    return image


def convert_img_to_byte_array(image: Image) -> bytes:
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="png")
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr


def get_sampler(sampler_name: str, model):
    if sampler_name == "DDIM" or sampler_name == "":
        sampler = DDIMSampler(model)
    elif sampler_name == "DPM":
        sampler = DPMSolverSampler(model)
    elif sampler_name == "PLMS":
        sampler = PLMSSampler(model)
    else:
        sampler = KSampler(model, sampler_name)
    return sampler


def parse_lora_array(arr: list[tuple[str, float]]) -> list[Lora]:
    lora_arr = []
    for file, value in arr:
        if file != "":
            lora_arr.append(Lora(os.path.join(LORAS_PATH, file), value))
    return lora_arr


def generate_txt2img(args: TextArgs) -> bytes:
    model = os.path.join(MODELS_PATH, args.model)
    vae = os.path.join(VAES_PATH, args.vae) if args.vae != "" else ""
    pipeline = DiffusionTxt2ImgPipeline(model, vae)
    sampler = get_sampler(args.sampler, pipeline.model)
    embedding = (
        Embedding(os.path.join(EMBEDDINGS_PATH, args.embedding))
        if args.embedding != ""
        else None
    )
    loras = parse_lora_array(args.loras)
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    steps = args.steps
    seed = args.seed
    scale = args.scale
    ddim_eta = args.ddim_eta
    H = args.height
    W = args.width
    layer_skip = args.layer_skip

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        steps=steps,
        seed=seed,
        scale=scale,
        ddim_eta=ddim_eta,
        sampler=sampler,
        H=H,
        W=W,
        layer_skip=layer_skip,
        loras=loras,
        embedding=embedding,
    )

    if args.upscale > 1:
        upscale_pipeline = DiffusionUpscalingPipeline(model, vae)
        image = upscale_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=image,
            upscale=args.upscale,
            scale=scale,
            ddim_eta=ddim_eta,
            layer_skip=layer_skip,
            loras=loras,
            embedding=embedding,
        )

    img_bytes = convert_img_to_byte_array(image)

    return img_bytes


def generate_img2img(args: ImageArgs) -> bytes:
    model = os.path.join(MODELS_PATH, args.model)
    vae = os.path.join(VAES_PATH, args.vae) if args.vae != "" else ""
    pipeline = DiffusionImg2ImgPipeline(model, vae)
    embedding = (
        Embedding(os.path.join(EMBEDDINGS_PATH, args.embedding))
        if args.embedding != ""
        else None
    )
    loras = parse_lora_array(args.loras)
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    init_image = convert_url_to_img(args.image)
    steps = args.steps
    seed = args.seed
    scale = args.scale
    strength = args.strength
    ddim_eta = args.ddim_eta
    layer_skip = args.layer_skip

    image = pipeline(
        prompt=prompt,
        negative_prompt=negative_prompt,
        init_image=init_image,
        steps=steps,
        seed=seed,
        scale=scale,
        strength=strength,
        ddim_eta=ddim_eta,
        layer_skip=layer_skip,
        loras=loras,
        embedding=embedding,
    )

    if args.upscale > 1:
        upscale_pipeline = DiffusionUpscalingPipeline(model, vae)
        image = upscale_pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            init_image=image,
            upscale=args.upscale,
            scale=scale,
            ddim_eta=ddim_eta,
            layer_skip=layer_skip,
            loras=loras,
            embedding=embedding,
        )

    img_bytes = convert_img_to_byte_array(image)

    return img_bytes
