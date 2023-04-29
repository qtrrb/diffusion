from pydantic import BaseModel


class TextArgs(BaseModel):
    prompt: str
    negative_prompt: str
    height: int = 512
    width: int = 512
    model: str = "sdv1/1-5.safetensors"
    vae: str = ""


class ImageArgs(BaseModel):
    prompt: str
    negative_prompt: str
    image: str
    model: str = "sdv1/1-5.safetensors"
    vae: str = ""
