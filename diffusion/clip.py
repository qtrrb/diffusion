import torch
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModel

import math


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer_idx=-1,
    ):  # clip-vit-base-patch32
        super().__init__()
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer_idx = layer_idx
        assert 1 <= abs(layer_idx) <= 12

    def freeze(self):
        self.transformer = self.transformer.eval()
        # self.train = disabled_train
        for param in self.parameters():
            param.requires_grad = False

    def old_forward(self, text):
        batch_encoding = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )
        tokens = batch_encoding["input_ids"].to(self.device)
        outputs = self.transformer(
            input_ids=tokens, output_hidden_states=(abs(self.layer_idx) > 1)
        )
        if abs(self.layer_idx) > 1:
            z = self.transformer.text_model.final_layer_norm(
                outputs.hidden_states[self.layer_idx]
            )
        else:
            z = outputs.last_hidden_state
        return z

    def forward(self, text):
        # Tokenize text
        batch_encoding = self.tokenizer(
            text,
            truncation=False,
            max_length=self.max_length,
            return_length=True,
            return_overflowing_tokens=False,
            padding="max_length",
            return_tensors="pt",
        )

        tokens = batch_encoding["input_ids"].to(self.device)
        chunks = math.ceil((tokens.shape[1] - 2) / (self.max_length - 2))
        chunk_embeddings = []
        for i in range(chunks):
            token_chunk = tokens[
                :, i * (self.max_length - 2) : (i + 1) * (self.max_length - 2) + 2
            ].clone()
            # Pad the chunk if its length is less than max_length
            if token_chunk.shape[1] < self.max_length:
                padding_length = self.max_length - token_chunk.shape[1]
                padding = torch.zeros(
                    (token_chunk.shape[0], padding_length), dtype=torch.int32
                ).to(self.device)
                padding[:, :] = self.tokenizer.pad_token_id
                token_chunk = torch.cat([token_chunk, padding], dim=1)

            # add starting and ending tokens
            token_chunk[:, 0] = tokens[:, 0]
            token_chunk[:, -1] = tokens[:, -1]

            outputs = self.transformer(
                input_ids=token_chunk, output_hidden_states=(abs(self.layer_idx) > 1)
            )
            if abs(self.layer_idx) > 1:
                chunk_embedding = self.transformer.text_model.final_layer_norm(
                    outputs.hidden_states[self.layer_idx]
                )
            else:
                chunk_embedding = outputs.last_hidden_state
            chunk_embeddings.append(chunk_embedding)

        z = torch.cat(chunk_embeddings, dim=1)
        return z


    def encode(self, text):
        return self(text)

