# Based on https://github.com/jordanramstad/InvokeAI/blob/add_lora_support/ldm/modules/legacy_lora_manager.py
# https://github.com/kohya-ss/sd-scripts/blob/main/networks/lora.py
import re
import torch
from safetensors.torch import load_file

re_digits = re.compile(r"\d+")
re_text_blocks = re.compile(r"lora_te_(.+)")
re_unet_down_blocks = re.compile(r"lora_unet_down_blocks_(\d+)_attentions_(\d+)_(.+)")
re_unet_mid_blocks = re.compile(r"lora_unet_mid_block_attentions_0_(.+)")
re_unet_up_blocks = re.compile(r"lora_unet_up_blocks_(\d+)_attentions_(\d+)_(.+)")

loaded_lora = [] 
toload_lora = [] 

# values from https://github.com/cloneofsimo/lora/blob/master/lora_diffusion/to_ckpt_v2.py
# and  convert code loosely based on https://github.com/jordanramstad/InvokeAI/blob/add_lora_support/ldm/modules/legacy_lora_manager.py
def convert_key_to_diffusers( key):
    def match(match_list, regex):
        r = re.match(regex, key)
        if not r:
            return False

        match_list.clear()
        match_list.extend([int(x) if re.match(re_digits, x) else x for x in r.groups()])
        return True

    m = []

    if match(m, re_text_blocks):
        return f"transformer_{m[0]}"

    if match(m, re_unet_down_blocks):
        return f"diffusion_model_input_blocks_{3 * m[0] + m[1] + 1}_1_{m[2]}"

    if match(m, re_unet_mid_blocks):
        return f"diffusion_model_middle_block_1_{m[0]}"

    if match(m, re_unet_up_blocks):
        return f"diffusion_model_output_blocks_{3 * m[0] + m[1]}_1_{m[2]}"

    return key

def create_mapping(model):
    mapping = {}
    for name, module in model.cond_stage_model.named_modules():
        if module.__class__.__name__ == "Linear" or (module.__class__.__name__ == "Conv2d" and module.kernel_size == (1, 1)):
            lora_name = name.replace('.', '_')
            mapping[lora_name] = module
            module.lora_layer_name = lora_name
    for name, module in model.model.named_modules():
        if module.__class__.__name__ == "Linear" or (module.__class__.__name__ == "Conv2d" and module.kernel_size == (1, 1)):
            lora_name = name.replace('.', '_')
            mapping[lora_name] = module
            module.lora_layer_name = lora_name
    return mapping

class LoRAModule:
    def __init__(self):
        self.up = None
        self.down = None
        self.alpha = None
        self.rank = None

class LoRA:
    def __init__(self, file, multiplier=1.0):
        self.file = file
        self.multiplier = multiplier
        self.modules = {}


    def load_lora(self, sd_model):
        global loaded_lora
        if self.file.endswith(".safetensors"):
            ckpt = load_file(self.file, device="cpu")
        else:
            ckpt = torch.load(self.file, map_location="cpu")
        
        if "state_dict" in ckpt:
            sd = ckpt["state_dict"]
        else:
            sd = ckpt
        for key, value in sd.items():
            key =convert_key_to_diffusers(key)
            stem, leaf = key.split(".", 1)

            mapping = create_mapping(sd_model)
            
            sd_module = mapping.get(stem, None)
            if sd_module is None:
                print(f">> Missing layer: {stem}")
                continue

            lora_module = self.modules.get(stem, None)
            if lora_module is None:
                lora_module = LoRAModule()
                self.modules[stem] = lora_module 
            
            if leaf.endswith("alpha"):
                if lora_module.alpha is None:
                    lora_module.alpha = value.item()
                continue

            if leaf == "lora_down.weight" and lora_module.rank is None:
                lora_module.rank = value.shape[0]
           
            if type(sd_module) == torch.nn.Linear:
                module = torch.nn.Linear(value.shape[1], value.shape[0], bias=False)
            elif type(sd_module) == torch.nn.Conv2d:
                module = torch.nn.Conv2d(value.shape[1], value.shape[0], (1, 1), bias=False)
            else:
                print(f">> Encountered unknown lora layer module: {type(value).__name__}")
                return
                
            with torch.no_grad():
                module.weight.copy_(value)

            module.to(device="cuda")


            if leaf == "lora_up.weight":
                lora_module.up = module
            elif leaf == "lora_down.weight":
                lora_module.down = module
            else:
                print(f">> Encountered unknown layer in: {leaf}")
                return
        loaded_lora.append(self)

    def __call__(self):
        global toload_lora
        toload_lora.append(self)

def apply_lora(sd_model):
    global toload_lora
    if toload_lora is not []:
        for lora in toload_lora:
            lora.load_lora(sd_model)
    toload_lora = []
    if loaded_lora is not []:
        if not hasattr(torch.nn.Linear, 'old_forward'):
            torch.nn.Linear.old_forward = torch.nn.Linear.forward
        if not hasattr(torch.nn.Conv2d, 'old_forward'):
            torch.nn.Conv2d.old_forward = torch.nn.Conv2d.forward 
        torch.nn.Linear.forward = Linear_forward
        torch.nn.Conv2d.forward = Conv2d_forward

def unload_lora():
    global toload_lora
    global loaded_lora
    toload_lora = []
    loaded_lora = []

def lora_forward(module, input, output):
    for lora in loaded_lora:
        layer = lora.modules.get(getattr(module, 'lora_layer_name', None), None)
        if layer is not None:
# It seems like for some LoRAs, alpha is None
            if layer.alpha is not None:
                output = output + layer.up(layer.down(input)) * lora.multiplier * (layer.alpha / layer.rank)
            else: 
                output = output + layer.up(layer.down(input)) * lora.multiplier
                

    return output 

def Linear_forward(self, input):
    return lora_forward(self, input,  torch.nn.Linear.old_forward(self, input))

def Conv2d_forward(self, input):
    return lora_forward(self, input, torch.nn.Conv2d.old_forward(self, input))


