import torch
from ..scripts.txt2imgapi import generate
from ..scripts.img2imgapi import generate_from_image
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from .schemas import TextArgs, ImageArgs

app = FastAPI()


@app.get("/")
def read_root():
    return {"Status": "API running"}


@app.post(
    "/txt2img", response_class=Response, responses={200: {"content": {"image/png": {}}}}
)
def txt2img(args: TextArgs):
    image_bytes: bytes = generate(
        prompt=args.prompt,
        negative_prompt=args.negative_prompt,
        W=args.width,
        H=args.height,
        ckpt=args.model,
        vae=args.vae,
    )
    torch.cuda.empty_cache()
    # Return the image in the response
    return Response(content=image_bytes, media_type="image/png")


@app.post(
    "/img2img", response_class=Response, responses={200: {"content": {"image/png": {}}}}
)
def img2img(args: ImageArgs):
    try:
        image_bytes: bytes = generate_from_image(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            init_img_url=args.image,
            ckpt=args.model,
            vae=args.vae,
        )
        torch.cuda.empty_cache()
        # Return the image in the response
        return Response(content=image_bytes, media_type="image/png")
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Error generating image")
