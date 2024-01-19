import numpy as np
import os
import PIL
import random
import torch
import typing

from einops import rearrange, repeat
from PIL import Image
from torch import autocast

from ..ldm.models.diffusion.ddim import DDIMSampler
from .diffusion_pipeline import DiffusionPipeline
from ..lora import Lora, LoRAManager
from ..textual_inversion import Embedding, TextualInversionManager


class DiffusionImg2ImgPipeline(DiffusionPipeline):
    def __init__(
        self,
        model_path: str | os.PathLike,
        vae_path: str | os.PathLike = "",
        version: typing.Literal["v1", "v2"] = "v1",
        dtype=torch.float16,
    ):
        super().__init__(model_path, vae_path, version, dtype)

    def load_img(self, image: PIL.Image):
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((w, h), resample=PIL.Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        return 2.0 * image - 1.0

    def generate(
        self,
        prompt="",
        negative_prompt="",
        init_image=None,
        steps=30,
        seed=-1,
        scale=7,
        strength=0.7,
        ddim_eta=0.0,
        batch_size=1,
        layer_skip=1,
        loras: list[Lora] = [],
        embedding: Embedding | None = None,
    ):
        assert prompt != ""
        assert init_image is not None

        sampler = DDIMSampler(self.model)

        if seed == -1:
            seed = random.randint(0, 9999999999)

        print(f"Seed set to {seed}")
        torch.manual_seed(seed)

        if self.version == "v1":
            self.model.cond_stage_model.layer_skip = layer_skip

        init_image_tensor = self.load_img(init_image).to(torch.device("cuda"))
        init_image_tensor = repeat(init_image_tensor, "1 ... -> b ...", b=batch_size)
        init_latent = self.model.get_first_stage_encoding(
            self.model.encode_first_stage(init_image_tensor)
        )  # move to latent space

        sampler.make_schedule(ddim_num_steps=steps, ddim_eta=ddim_eta, verbose=False)
        assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
        t_enc = int(strength * steps)
        print(f"target t_enc is {t_enc} steps")

        lora_manager = LoRAManager(loras)
        lora_manager.load_loras(self.model)

        if embedding is not None:
            textual_inversion_manager = TextualInversionManager(self.model, embedding)
            textual_inversion_manager.apply_textual_inversion_embeddings()
            prompt = textual_inversion_manager.replace_token_in_prompt(prompt)
            negative_prompt = textual_inversion_manager.replace_token_in_prompt(
                negative_prompt
            )

        precision_scope = autocast

        with torch.no_grad(), precision_scope("cuda"), self.model.ema_scope():
            uc = self.model.get_learned_conditioning(batch_size * [negative_prompt])
            c = self.model.get_learned_conditioning(batch_size * [prompt])

            # encode (scaled latent)
            z_enc = sampler.stochastic_encode(
                init_latent, torch.tensor([t_enc] * batch_size).to(torch.device("cuda"))
            )
            # decode it
            samples = sampler.decode(
                z_enc,
                c,
                t_enc,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=uc,
            )

            imgs = []
            x_samples = self.model.decode_first_stage(samples)
            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            for x_sample in x_samples:
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
        init_image=None,
        steps=30,
        seed=-1,
        scale=7,
        strength=0.7,
        ddim_eta=0.0,
        batch_size=1,
        layer_skip=1,
        loras: list[Lora] = [],
        embedding: Embedding | None = None,
    ):
        return self.generate(
            prompt,
            negative_prompt,
            init_image,
            steps,
            seed,
            scale,
            strength,
            ddim_eta,
            batch_size,
            layer_skip,
            loras,
            embedding,
        )
