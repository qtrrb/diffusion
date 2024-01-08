# qtrrb/diffusion

A fork of [stable diffusion](https://github.com/Stability-AI/stablediffusion) that adds optimizations, features and an API

![hero](.github/assets/banner.png)

## THIS PROJECT WILL BE CONTINUED SOON

TODO: 

- Refactor code to be closer to the original structure
- Remove SD2 support for now, until all 1.5 features are supported, Focus for now is on 1.5
- Refactor K-Diffusion schedulers to add more
- Improve LoRA and VAE loading
- Cleanup pipelines
- Improve API

## Main Features

- Upscaling
- A1111 LoRA support
- Textual inversion support
- ControlNet
- Unlimited tokens
- VAE loading
- K-Diffusion schedulers
- Safetensors support
- Optimizations (fp16, Attention slicing)
- Text to Image API
- Image to Image API

## Usage

- Clone this repository

```bash
git clone https://github.com/qtrrb/diffusion
```

- Initialize a new virtual environment

```bash
python -m venv .env
```

- Activate the virtual environment

- Install necessary dependencies

```bash
pip install -r requirements.txt
```

- Run the API with

```bash
python -m diffusion
```
