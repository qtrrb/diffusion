import argparse, os
import cv2
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from itertools import islice
from einops import rearrange
from torchvision.utils import make_grid
from pytorch_lightning import seed_everything
from torch import autocast, float32
from contextlib import nullcontext
from imwatermark import WatermarkEncoder

from core.ldm.util import instantiate_from_config
from core.ldm.models.diffusion.ddim import DDIMSampler
from core.ldm.models.diffusion.plms import PLMSSampler
from core.ldm.models.diffusion.dpm_solver import DPMSolverSampler

from safetensors.torch import load_file

import random
import io

from transformers import logging

logging.set_verbosity_error()
torch.set_grad_enabled(False)

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

    #this is done after model.half() to avoid  converting VAE weights to float16
    if  vae:
        vae = "core/ldm/models/" + vae
        vae_sd = torch.load(vae, map_location="cpu")["state_dict"]
        model.first_stage_model.load_state_dict(vae_sd,  strict=False)

    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img
        ), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def generate(
    prompt="",
    negative_prompt="",
    steps=30,
    seed="random",
    ckpt="sdv1/1-5.safetensors",
    vae="",
    precision="autocast",
    scale=7,
    ddim_eta=0.0,
    plms=False,
    dpm=False,
    H=512,
    W=512,
    C=4, #Latent Channels
    f=8, #downsampling factor
    ):
    if "sdv1" in ckpt:
        config_file="core/configs/stable-diffusion/v1-inference.yaml"
    else:
        config_file="core/configs/stable-diffusion/v2-inference.yaml"

    ckpt = "core/ldm/models/" + ckpt

    if seed == "random":
        seed = random.randint(0,999999)
    seed_everything(seed)

    config = OmegaConf.load(f"{config_file}")
    model = load_model_from_config(config, f"{ckpt}", f"{vae}")
    
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    model = model.to(device)

    if plms:
        sampler = PLMSSampler(model)
    elif dpm:
        sampler = DPMSolverSampler(model)
    else:
        sampler = DDIMSampler(model)

#    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
#    wm = "SDV2"
#    wm_encoder = WatermarkEncoder()
#    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    assert prompt is not None
    start_code = None
    
    # might remove but generally makes results better
    negative_prompt += "nsfw, lowres, bad anatomy, bad hands, text, missing finger, extra digits, fewer digits, blurry, mutated hands and fingers, poorly drawn face, mutation, deformed face, ugly, bad proportions, extra limbs, extra face, double head, extra head, extra feet, monster, logo, cropped, worst quality, low quality, normal quality, jpeg, humpbacked, long body, long neck, jpeg artifacts"

    precision_scope = autocast if precision == "autocast" else nullcontext
    with torch.no_grad(), \
        precision_scope("cuda"), \
        model.ema_scope():
            all_samples = list()

            uc = model.get_learned_conditioning([negative_prompt])
            c = model.get_learned_conditioning([prompt])
            shape = [C, H // f, W // f]
            samples, _ = sampler.sample(S=steps,
                                                conditioning=c,
                                                batch_size=1,
                                                shape=shape,
                                                verbose=False,
                                                unconditional_guidance_scale=scale,
                                                unconditional_conditioning=uc,
                                                eta=ddim_eta,
                                                x_T=start_code)

            x_sample = model.decode_first_stage(samples)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

            x_sample = 255. * rearrange(x_sample.cpu().numpy(), '1 c h w -> h w c')
            img = Image.fromarray(x_sample.astype(np.uint8))
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='png')
            img_byte_arr = img_byte_arr.getvalue()

            all_samples.append(x_sample)


    print("Done!")

    return img_byte_arr