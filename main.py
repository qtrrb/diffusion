import uvicorn
import torch
from core.scripts.txt2imgapi import generate
from core.scripts.img2imgapi import generate_from_image
from fastapi import FastAPI, HTTPException
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
    responses={
        200: {
            "content": {"image/png": {}}
        }
    }

)
def txt2img(prompt: Prompt):
    try:
        image_bytes: bytes = generate(prompt=prompt.prompt, negative_prompt=prompt.negative_prompt,
                                      W=prompt.width, H=prompt.height, ckpt=prompt.model, vae=prompt.vae)
        torch.cuda.empty_cache()
        # Return the image in the response
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error generating image")

@app.post(
    "/img2img",
    response_class=Response,
    responses={
        200: {
            "content": {"image/png": {}}
        }
    }

)
def img2img(prompt: ImagePrompt):
    try:
        image_bytes: bytes = generate_from_image(prompt=prompt.prompt, negative_prompt=prompt.negative_prompt, 
                                                 init_img_url=prompt.image, ckpt=prompt.model, vae=prompt.vae)
        torch.cuda.empty_cache()
        # Return the image in the response
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error generating image")


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")
