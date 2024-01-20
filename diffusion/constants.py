import os

CONSTANTS_PATH = os.path.abspath(os.path.dirname(__file__))
CONFIGS_PATH = os.path.join(CONSTANTS_PATH, "./configs/")
MODELS_PATH = os.path.join(CONSTANTS_PATH, "./../models/sdv1/")
LORAS_PATH = os.path.join(CONSTANTS_PATH, "./../models/lora/")
VAES_PATH = os.path.join(CONSTANTS_PATH, "./../models/vae/")
EMBEDDINGS_PATH = os.path.join(CONSTANTS_PATH, "./../models/embedding/")
CONTROL_MODELS_PATH = os.path.join(CONSTANTS_PATH, "./../models/controlnet/")
