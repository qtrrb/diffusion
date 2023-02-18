from core.scripts.txt2imgapi import generate
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

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post(
    "/txt2img", 
    response_class=Response,
    responses = {
        200: {
            "content": {"image/png": {}}
        }
    }

)
def generateImage(prompt: Prompt):
    image_bytes: bytes = generate(prompt=prompt.prompt,negative_prompt=prompt.negative_prompt,W=prompt.width, H=prompt.height, ckpt="core/ldm/models/" + prompt.model,)
    # Return the image in the response
    return Response(content=image_bytes, media_type="image/png")