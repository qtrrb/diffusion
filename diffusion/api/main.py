import os
import torch
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TextArgs, ImageArgs
from .util import generate_txt2img, generate_img2img, get_files
from ..constants import MODELS_PATH, VAES_PATH, EMBEDDINGS_PATH, LORAS_PATH
from transformers import logging

logging.set_verbosity_error()

busy = False
app = FastAPI(title="diffusion api")
router = APIRouter(prefix="/v1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "API running"}


@router.get("/models")
def get_models() -> list[str]:
    models_list = get_files(os.path.join(MODELS_PATH, "sdv1"))
    for i in range(len(models_list)):
        models_list[i] = "sdv1/" + models_list[i]
    return models_list


@router.get("/vaes")
def get_vaes() -> list[str]:
    return get_files(VAES_PATH)


@router.get("/embeddings")
def get_embeddings() -> list[str]:
    return get_files(EMBEDDINGS_PATH)


@router.get("/loras")
def get_loras() -> list[str]:
    return get_files(LORAS_PATH)


@router.post(
    "/generate",
    response_class=Response,
    responses={
        200: {"content": {"image/png": {}}},
        500: {
            "description": "Error generating image",
            "content": {
                "application/json": {"example": {"detail": "Error generating image"}}
            },
        },
        429: {
            "description": "Model is overloaded",
            "content": {
                "application/json": {"example": {"detail": "Model is overloaded"}}
            },
        },
    },
)
def generate_image(args: TextArgs):
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
        500: {
            "description": "Error generating image",
            "content": {
                "application/json": {"example": {"detail": "Error generating image"}}
            },
        },
        429: {
            "description": "Model is overloaded",
            "content": {
                "application/json": {"example": {"detail": "Model is overloaded"}}
            },
        },
    },
)
def vary_image(args: ImageArgs):
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
