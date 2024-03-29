import numpy as np
import os
import random
import torch
import typing

from einops import rearrange
from PIL import Image
from torch import autocast

from ..ldm.models.diffusion.ddim import DDIMSampler
from .diffusion_pipeline import DiffusionPipeline
from ..lora import Lora, LoRAManager
from ..textual_inversion import TextualInversionManager


class DiffusionTxt2ImgPipeline(DiffusionPipeline):
    def __init__(
        self,
        model_path: str | os.PathLike,
        vae_path: str | os.PathLike = "",
        dtype=torch.float16,
    ):
        super().__init__(model_path, vae_path, dtype)

    def generate(
        self,
        prompt="",
        negative_prompt="",
        steps=30,
        seed=-1,
        scale=7,
        ddim_eta=0.0,
        sampler=None,
        batch_size=1,
        H=512,
        W=512,
        layer_skip=1,
        loras: list[Lora] = [],
    ):
        assert prompt != ""

        start_code = None
        C = 4  # Latent Channels
        f = 8  # downsampling factor

        if sampler is None:
            sampler = DDIMSampler(self.model)

        if seed == -1:
            seed = random.randint(0, 9999999999)

        print(f"Seed set to {seed}")
        torch.manual_seed(seed)

        self.model.cond_stage_model.layer_idx = -layer_skip

        lora_manager = LoRAManager(loras)
        lora_manager.load_loras(self.model)

        textual_inversion_manager = TextualInversionManager(self.model)

        prompt = textual_inversion_manager.process_prompt(prompt)
        negative_prompt = textual_inversion_manager.process_prompt(
                negative_prompt
        )

        precision_scope = autocast

        with torch.no_grad(), precision_scope("cuda"), self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [negative_prompt])
            c = self.model.get_learned_conditioning(batch_size * [prompt])
            shape = [C, H // f, W // f]
            samples, _ = sampler.sample(
                S=steps,
                conditioning=c,
                batch_size=batch_size,
                shape=shape,
                verbose=False,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
                eta=ddim_eta,
                x_T=start_code,
            )

            x_samples = self.model.decode_first_stage(samples)
            imgs = []
            for x_sample in x_samples:
                x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)

                x_sample = 255.0 * rearrange(x_sample.cpu().numpy(), "c h w -> h w c")
                img = Image.fromarray(x_sample.astype(np.uint8))
                imgs.append(img)

        lora_manager.clear_loras()
        print("Done!")

        return imgs

    def __call__(
        self,
        prompt="",
        negative_prompt="",
        steps=30,
        seed=-1,
        scale=7,
        ddim_eta=0.0,
        sampler=None,
        batch_size=1,
        H=512,
        W=512,
        layer_skip=1,
        loras: list[Lora] = [],
    ):
        return self.generate(
            prompt,
            negative_prompt,
            steps,
            seed,
            scale,
            ddim_eta,
            sampler,
            batch_size,
            H,
            W,
            layer_skip,
            loras,
        )
