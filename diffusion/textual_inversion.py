import os
import safetensors
import torch
from pathlib import Path
from typing import List

from .constants import EMBEDDINGS_PATH

class Embedding:
    def __init__(self, file_path: str | os.PathLike):
        self.file_path = file_path
        self.token, self.vecs = self.load_textual_inversion_embedding()

    def load_textual_inversion_embedding(self):
        if self.file_path.endswith(".safetensors"):
            state_dict = safetensors.torch.load_file(self.file_path, device="cpu")
        else:
            state_dict = torch.load(self.file_path, map_location="cpu")
        if "string_to_param" in state_dict:
            vecs = state_dict["string_to_param"]["*"]
        else:
            vecs = state_dict["emb_params"]
        token = Path(self.file_path).stem

        print(f"Using textual inversion embedding with token {token}")

        return token, vecs


class TextualInversionManager:
    def __init__(self, model):
        self.model = model
        self.tokenizer = model.cond_stage_model.tokenizer
        self.text_encoder = model.cond_stage_model.transformer
        self.embeddings = self.load_embeddings_from_folder(EMBEDDINGS_PATH)

    def load_embeddings_from_folder(self, folder_path):
        embeddings = []
        for file_name in os.listdir(folder_path):
            if file_name == ".gitkeep":
                continue
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                embeddings.append(Embedding(file_path))
        return embeddings
    
    def process_prompt(self, prompt: str):
        for embedding in self.embeddings:
            if embedding.token in prompt:
                print(f"{embedding.token} in prompt, using embedding...")
                # Apply the embedding
                for i, emb in enumerate(embedding.vecs):
                    token_name = f"{embedding.token}_{i}"
                    self.tokenizer.add_tokens([token_name])
                    token_id = self.tokenizer.convert_tokens_to_ids(token_name)
                    emb = emb.to(dtype=self.text_encoder.dtype, device=self.text_encoder.device)
                    self.text_encoder.resize_token_embeddings(len(self.tokenizer))
                    self.text_encoder.get_input_embeddings().weight.data[token_id] = emb

                # Replace tokens in prompt
                token_index = prompt.index(embedding.token)
                token_replacement = " ".join(
                    [
                        f"{embedding.token}_{i}"
                        for i in range(embedding.vecs.size()[0])
                    ]
                )
                prompt = (
                    prompt[:token_index]
                    + token_replacement
                    + prompt[token_index + len(embedding.token):]
                )
        return prompt