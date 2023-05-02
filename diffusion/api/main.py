import torch
from fastapi import FastAPI, APIRouter, HTTPException
from fastapi.responses import Response
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TextArgs, ImageArgs
from ..utils.api_util import generate_txt2img, generate_img2img
from transformers import logging

logging.set_verbosity_error()

app = FastAPI(title="diffusion api")
router = APIRouter(prefix="/v1/images")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@router.get("/")
def read_root():
    return {"status": "API running"}


@router.post(
    "/generate",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def txt2img(args: TextArgs):
    try:
        image_bytes: bytes = generate_txt2img(args=args)
        torch.cuda.empty_cache()
        # Return the image in the response
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error generating image")


@router.post(
    "/vary",
    response_class=Response,
    responses={200: {"content": {"image/png": {}}}},
)
def img2img(args: ImageArgs):
    try:
        image_bytes: bytes = generate_img2img(args)
        torch.cuda.empty_cache()
        # Return the image in the response
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error generating image")


app.include_router(router)
