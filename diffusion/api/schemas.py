from pydantic import BaseModel


class TextArgs(BaseModel):
    model: str = "sdv1/1-5.safetensors"
    vae: str = ""
    prompt: str
    negative_prompt: str = ""
    steps: int = 30
    seed: int = -1
    scale: int = 7
    ddim_eta: float = 0.0
    sampler: str = ""
    height: int = 512
    width: int = 512
    layer_skip: int = 1
    loras: list[str] = []
    embedding: str = ""


class ImageArgs(BaseModel):
    model: str = "sdv1/1-5.safetensors"
    vae: str = ""
    prompt: str
    negative_prompt: str = ""
    image: str = ""
    steps: int = 30
    seed: int = -1
    scale: int = 7
    ddim_eta: float = 0.0
    layer_skip: int = 1
    loras: list[str] = []
    embedding: str = ""
