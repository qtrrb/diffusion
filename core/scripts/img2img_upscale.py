"""
Upscales image by loading it, multiplying dimensions by 2 and then applying the diffusion process
TODO: Every file should be rewritten as a class and return an image so this could be applied to any 
img2img or txt2img process
"""

import argparse
import os
import io
import random
import requests
import PIL
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange, repeat
from torchvision.utils import make_grid
from torch import autocast
from contextlib import nullcontext
from pytorch_lightning import seed_everything
from imwatermark import WatermarkEncoder
from safetensors.torch import load_file

from core.scripts.txt2img import put_watermark
from core.ldm.util import instantiate_from_config
from core.ldm.models.diffusion.ddim import DDIMSampler


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, vae, verbose=False):
    print(f"Loading model from {ckpt}")
    if ckpt.endswith("safetensors"):
        pl_sd = load_file(ckpt, device="cpu")
    else:
        pl_sd = torch.load(ckpt, map_location="cpu")

    if "state_dict" in pl_sd:
        sd = pl_sd["state_dict"]
    else:
        sd = pl_sd

    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    model.half()

    # this is done after model.half() to avoid  converting VAE weights to float16
    if vae:
        vae = "core/ldm/models/" + vae
        vae_sd = torch.load(vae, map_location="cpu")["state_dict"]
        model.first_stage_model.load_state_dict(vae_sd,  strict=False)

    return model


def load_img(path):
    image = Image.open(path).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {path}")
    # resize to integer multiple of 64
    w, h = map(lambda x: x - x % 64, (w, h))
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def load_img_from_url(url):
    response = requests.get(url)
    image = Image.open(io.BytesIO(response.content)).convert("RGB")
    w, h = image.size
    print(f"loaded input image of size ({w}, {h}) from {url}")
    # resize to integer multiple of 64
    w, h = map(lambda x: x * 2 - x % 64, (w, h))
#    w, h = map(lambda x: 832 if x > 832 else x, (w, h)) # limit size to 832 to avoid memory issues
    image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
    image = np.array(image).astype(np.float32) / 255.0
    image = image[None].transpose(0, 3, 1, 2)
    image = torch.from_numpy(image)
    return 2. * image - 1.


def generate_from_image(
    prompt="",
    negative_prompt="",
    init_img_url="",
    steps=50,
    seed="random",
    ckpt="sdv1/1-5.safetensors",
    vae="",
    precision="autocast",
    scale=9.0,
    strength=0.7,
    ddim_eta=0.0,
):
    if "sdv1" in ckpt:
        config_file = "core/configs/stable-diffusion/v1-inference.yaml"
    else:
        config_file = "core/configs/stable-diffusion/v2-inference.yaml"

    ckpt = "core/ldm/models/" + ckpt

    if seed == "random":
        seed = random.randint(0, 999999)
    seed_everything(seed)

    config = OmegaConf.load(f"{config_file}")
    model = load_model_from_config(config, f"{ckpt}", f"{vae}")

    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    sampler = DDIMSampler(model)

#    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
#    wm = "SDV2"
#    wm_encoder = WatermarkEncoder()
#    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    assert prompt is not None

    init_image = load_img_from_url(init_img_url).to(device)
    init_latent = model.get_first_stage_encoding(
        model.encode_first_stage(init_image))  # move to latent space

    sampler.make_schedule(ddim_num_steps=steps,
                          ddim_eta=ddim_eta, verbose=False)

    assert 0. <= strength <= 1., 'can only work with strength in [0.0, 1.0]'
    t_enc = int(strength * steps)
    print(f"target t_enc is {t_enc} steps")

    negative_prompt += "nsfw, lowres, bad anatomy, bad hands, text, missing finger, extra digits, fewer digits, blurry, mutated hands and fingers, poorly drawn face, mutation, deformed face, ugly, bad proportions, extra limbs, extra face, double head, extra head, extra feet, monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, jpeg artifacts"

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad(), \
            precision_scope("cuda"), \
            model.ema_scope():

        uc = model.get_learned_conditioning([negative_prompt])
        c = model.get_learned_conditioning([prompt])

        # encode (scaled latent)
        z_enc = sampler.stochastic_encode(
            init_latent, torch.tensor([t_enc]).to(device))
        # decode it
        samples = sampler.decode(z_enc, c, t_enc, unconditional_guidance_scale=scale,
                                 unconditional_conditioning=uc, )

        x_sample = model.decode_first_stage(samples)
        x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

        x_sample = 255. * rearrange(x_sample.cpu().numpy(), '1 c h w -> h w c')
        img = Image.fromarray(x_sample.astype(np.uint8))
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='png')
        img_byte_arr = img_byte_arr.getvalue()

    print("Done!")

    return img_byte_arr
