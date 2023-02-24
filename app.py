from core.scripts.txt2imgapi import generate
from core.scripts.img2imgapi import generate_from_image
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

app = FastAPI()

class Prompt(BaseModel):
    prompt: str
    negative_prompt: str
    height: int = 512
    width: int = 512
    model: str = "sdv1/1-5.safetensors"
    vae: str = ""

class ImagePrompt(BaseModel):
    prompt: str
    negative_prompt: str
    image: str
    model: str = "sdv1/1-5.safetensors"
    vae: str = ""

@app.get("/")
def read_root():
    return {"Status": "API running"}

@app.post(
    "/txt2img", 
    response_class=Response,
    responses = {
        200: {
            "content": {"image/png": {}}
        }
    }

)
def txt2img(prompt: Prompt):
    image_bytes: bytes = generate(prompt=prompt.prompt,negative_prompt=prompt.negative_prompt,W=prompt.width, H=prompt.height, ckpt=prompt.model, vae=prompt.vae)
    # Return the image in the response
    return Response(content=image_bytes, media_type="image/png")

@app.post(
    "/img2img", 
    response_class=Response,
    responses = {
        200: {
            "content": {"image/png": {}}
        }
    }

)
def img2img(prompt: ImagePrompt):
    image_bytes: bytes = generate_from_image(prompt=prompt.prompt,negative_prompt=prompt.negative_prompt,init_img_url=prompt.image, ckpt=prompt.model, vae=prompt.vae)
    # Return the image in the response
    return Response(content=image_bytes, media_type="image/png")
