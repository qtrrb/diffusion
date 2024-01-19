import numpy as np
import os
import random
import torch

from einops import rearrange, repeat
from omegaconf import OmegaConf
from PIL import Image
from safetensors.torch import load_file
from torch import autocast

from ..samplers.ddim import DDIMSampler
from .diffusion_pipeline import DiffusionPipeline
from ..modules.lora import Lora, LoRAManager
from ..modules.textual_inversion import Embedding, TextualInversionManager
from ..utils.constants import CONFIGS_PATH
from ..utils.util import instantiate_from_config


class DiffusionControlNetPipeline(DiffusionPipeline):
    def __init__(
        self,
        control_mode: str,
        control_model_path: str | os.PathLike,
        model_path: str | os.PathLike,
        vae_path: str | os.PathLike = "",
        dtype=torch.float16,
    ):
        self.control_mode = control_mode
        self.control_model_path = control_model_path
        super().__init__(model_path, vae_path, "v1", dtype)

    def load_model_and_vae(self, verbose=False):
        config_path = os.path.join(
            CONFIGS_PATH, "control_v11p_sd15.yaml"
        )
        config = OmegaConf.load(f"{config_path}")
        model = instantiate_from_config(config.model)
        print(f"Loading ControlNet model from {self.control_model_path}")
        if self.control_model_path.endswith("safetensors"):
            sd = load_file(self.control_model_path, device="cpu")
        else:
            sd = torch.load(self.control_model_path, map_location="cpu")

        if "state_dict" in sd:
            sd = sd["state_dict"]
        else:
            sd = sd

        model.load_state_dict(sd, strict=False)

        print(f"Loading model from {self.model_path}")
        if self.model_path.endswith("safetensors"):
            pl_sd = load_file(self.model_path, device="cpu")
        else:
            pl_sd = torch.load(self.model_path, map_location="cpu")

        if "state_dict" in pl_sd:
            sd = pl_sd["state_dict"]
        else:
            sd = pl_sd

        if "global_step" in pl_sd:
            print(f"Global Step: {pl_sd['global_step']}")
        m, u = model.load_state_dict(sd, strict=False)
        if len(m) > 0 and verbose:
            print("missing keys:")
            print(m)
        if len(u) > 0 and verbose:
            print("unexpected keys:")
            print(u)

        model.eval()

        # this is done after model.half() to avoid  converting VAE weights to float16
        if self.vae_path != "":
            print(f"Loading VAE from {self.vae_path}")
            if self.vae_path.endswith(".safetensors"):
                vae_sd = load_file(self.vae_path, device="cpu")
            else:
                vae_sd = torch.load(self.vae_path, map_location="cpu")["state_dict"]
            model.first_stage_model.load_state_dict(vae_sd, strict=False)

        assert torch.cuda.is_available(), "CUDA unavailable"
        device = torch.device("cuda")

        return model.to(device)

    def load_control_image(self, image: Image):
        w, h = image.size
        print(f"loaded input image of size ({w}, {h})")
        w, h = map(lambda x: x - x % 64, (w, h))  # resize to integer multiple of 64
        image = image.resize((w, h), resample=Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 255.0
        image = image[None].transpose(0, 3, 1, 2)
        image = torch.from_numpy(image)
        image = image.to("cuda")
        return image

    def generate(
        self,
        prompt="",
        negative_prompt="",
        init_image=None,
        steps=30,
        seed=-1,
        scale=7,
        strength=0.7,
        batch_size=1,
        ddim_eta=0.0,
        layer_skip=1,
        loras: list[Lora] = [],
        embedding: Embedding | None = None,
    ):
        assert prompt != ""
        assert init_image is not None
        C = 4  # Latent Channels
        f = 8  # downsampling factor

        sampler = DDIMSampler(self.model)

        if seed == -1:
            seed = random.randint(0, 9999999999)

        print(f"Seed set to {seed}")
        torch.manual_seed(seed)

        if self.version == "v1":
            self.model.cond_stage_model.layer_skip = layer_skip

        control = self.load_control_image(init_image)
        _, H, W = control[0].size()
        control = repeat(control, "1 ... -> b ...", b=batch_size)

        assert 0.0 <= strength <= 1.0, "can only work with strength in [0.0, 1.0]"
        self.model.control_scales = [strength] * 13

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
            cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning([prompt] * batch_size)
                ],
            }
            un_cond = {
                "c_concat": [control],
                "c_crossattn": [
                    self.model.get_learned_conditioning([negative_prompt] * batch_size)
                ],
            }

            shape = [C, H // f, W // f]
            samples, _ = sampler.sample(
                S=steps,
                shape=shape,
                batch_size=batch_size,
                conditioning=cond,
                verbose=False,
                eta=ddim_eta,
                unconditional_guidance_scale=scale,
                unconditional_conditioning=un_cond,
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
        batch_size=1,
        ddim_eta=0.0,
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
            batch_size,
            ddim_eta,
            layer_skip,
            loras,
            embedding,
        )
