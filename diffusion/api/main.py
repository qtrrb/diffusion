import torch
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TextArgs, ImageArgs
from ..utils.api_util import generate_txt2img, generate_img2img
from transformers import logging

logging.set_verbosity_error()

busy = False
app = FastAPI(title="diffusion api")
router = APIRouter(prefix="/v1/images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "API running"}


@router.post(
    "/generate",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        500: {"description": "Error generating image"},
        429: {"description": "Model is overloaded"},
    },
)
def txt2img(args: TextArgs):
    global busy
    if busy:
        raise HTTPException(status_code=429, detail="Model is overloaded")
    else:
        try:
            busy = True
            image_bytes: bytes = generate_txt2img(args=args)
            torch.cuda.empty_cache()
            busy = False
            # Return the image in the response
            return Response(content=image_bytes, media_type="image/png")
        except Exception as e:
            busy = False
            print(e)
            raise HTTPException(status_code=500, detail="Error generating image")


@router.post(
    "/vary",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        500: {"description": "Error generating image"},
        429: {"description": "Model is overloaded"},
    },
)
def img2img(args: ImageArgs):
    global busy
    if busy:
        raise HTTPException(status_code=429, detail="Model is overloaded")
    else:
        try:
            busy = True
            image_bytes: bytes = generate_img2img(args)
            torch.cuda.empty_cache()
            busy = False
            # Return the image in the response
            return Response(content=image_bytes, media_type="image/png")
        except Exception as e:
            busy = True
            print(e)
            raise HTTPException(status_code=500, detail="Error generating image")


app.include_router(router)
