import os
import torch
import typing
from abc import ABC
from omegaconf import OmegaConf
from safetensors.torch import load_file
from ..utils.constants import CONFIGS_PATH
from ..utils.util import instantiate_from_config


class DiffusionPipeline(ABC):
    def __init__(
        self,
        model_path: str | os.PathLike,
        vae_path: str | os.PathLike = "",
        version: typing.Literal["v1", "v2"] = "v1",
        dtype=torch.float16,
    ):
        self.model_path = model_path
        self.vae_path = vae_path
        self.version = version
        self.dtype = dtype
        self.model = self.load_model_and_vae()

    def load_model_and_vae(self, verbose=False):
        if self.version == "v1":
            config_file = os.path.join(CONFIGS_PATH, "v1-inference.yaml")
        else:
            config_file = os.path.join(CONFIGS_PATH, "v2-inference.yaml")

        config = OmegaConf.load(f"{config_file}")

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
        model.to(self.dtype)

        # this is done after model.half() to avoid  converting VAE weights to float16
        if self.vae_path != "":
            if self.vae_path.endswith(".safetensors"):
                vae_sd = load_file(self.vae_path, device="cpu")
            else:
                vae_sd = torch.load(self.vae_path, map_location="cpu")["state_dict"]
            model.first_stage_model.load_state_dict(vae_sd, strict=False)

        assert torch.cuda.is_available(), "CUDA unavailable"
        device = torch.device("cuda")

        return model.to(device)
