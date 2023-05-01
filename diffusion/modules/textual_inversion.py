import os
import torch
from pathlib import Path


class Embedding:
    def __init__(self, file_path: str | os.PathLike):
        self.file_path = file_path
        self.token, self.vecs = self.load_textual_inversion_embedding()

    def load_textual_inversion_embedding(self):
        state_dict = torch.load(self.file_path, map_location="cpu")
        if "string_to_param" in state_dict:
            vecs = state_dict["string_to_param"]["*"]
        else:
            vecs = state_dict["emb_params"]
        token = Path(self.file_path).stem

        return token, vecs


class TextualInversionManager:
    def __init__(self, model, embedding: Embedding):
        self.model = model
        self.tokenizer = model.cond_stage_model.tokenizer
        self.text_encoder = model.cond_stage_model.transformer
        self.embedding = embedding

    def apply_textual_inversion_embeddings(self):
        for i, emb in enumerate(self.embedding.vecs):
            self.tokenizer.add_tokens([f"{self.embedding.token}_{i}"])
            token_id = self.tokenizer.convert_tokens_to_ids(
                f"{self.embedding.token}_{i}"
            )
            emb = emb.to(dtype=self.text_encoder.dtype, device=self.text_encoder.device)
            self.text_encoder.resize_token_embeddings(len(self.tokenizer))
            self.text_encoder.get_input_embeddings().weight.data[token_id] = emb

    def replace_token_in_prompt(self, prompt: str):
        if self.embedding.token in prompt:
            token_index = prompt.index(self.embedding.token)
            token_replacement = " ".join(
                [
                    f"{self.embedding.token}_{i}"
                    for i in range(self.embedding.vecs.size()[0])
                ]
            )
            return (
                prompt[:token_index]
                + token_replacement
                + prompt[token_index + len(self.embedding.token) :]
            )
        else:
            return prompt
