import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from transformers import CLIPTokenizer, CLIPTextModel

import open_clip

import math


class AbstractEncoder(nn.Module):
    def __init__(self):
        super().__init__()

    def encode(self, *args, **kwargs):
        raise NotImplementedError


class FrozenCLIPEmbedder(AbstractEncoder):
    """Uses the CLIP transformer encoder for text (from huggingface)"""

    LAYERS = ["last", "pooled", "hidden"]

    def __init__(
        self,
        version="openai/clip-vit-large-patch14",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
        layer_idx=None,
        layer_skip=1,
    ):  # clip-vit-base-patch32
        super().__init__()
        assert layer in self.LAYERS
        self.tokenizer = CLIPTokenizer.from_pretrained(version)
        self.transformer = CLIPTextModel.from_pretrained(version)
        self.device = device
        self.max_length = max_length
        self.layer_skip = layer_skip
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = layer_idx
        if layer == "hidden":
            assert layer_idx is not None
            assert 0 <= abs(layer_idx) <= 12

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
            input_ids=tokens, output_hidden_states=(self.layer_skip > 1)
        )
        if self.layer_skip > 1:
            z = self.transformer.text_model.final_layer_norm(
                outputs.hidden_states[-self.layer_skip]
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
        if chunks > 1:
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
                    input_ids=token_chunk, output_hidden_states=(self.layer_skip > 1)
                )
                if self.layer_skip > 1:
                    chunk_embedding = self.transformer.text_model.final_layer_norm(
                        outputs.hidden_states[-self.layer_skip]
                    )
                else:
                    chunk_embedding = outputs.last_hidden_state
                chunk_embeddings.append(chunk_embedding)

            z = torch.cat(chunk_embeddings, dim=1)
            return z
        else:
            outputs = self.transformer(
                input_ids=tokens, output_hidden_states=(self.layer_skip > 1)
            )
            if self.layer_skip > 1:
                z = self.transformer.text_model.final_layer_norm(
                    outputs.hidden_states[-self.layer_skip]
                )
            else:
                z = outputs.last_hidden_state
        return z

    def encode(self, text):
        return self(text)


class FrozenOpenCLIPEmbedder(AbstractEncoder):
    """
    Uses the OpenCLIP transformer encoder for text
    """

    LAYERS = [
        # "pooled",
        "last",
        "penultimate",
    ]

    def __init__(
        self,
        arch="ViT-H-14",
        version="laion2b_s32b_b79k",
        device="cuda",
        max_length=77,
        freeze=True,
        layer="last",
    ):
        super().__init__()
        assert layer in self.LAYERS
        model, _, _ = open_clip.create_model_and_transforms(
            arch, device=torch.device("cpu"), pretrained=version
        )
        del model.visual
        self.model = model

        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        if self.layer == "last":
            self.layer_idx = 0
        elif self.layer == "penultimate":
            self.layer_idx = 1
        else:
            raise NotImplementedError()

    def freeze(self):
        self.model = self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, text):
        tokens = open_clip.tokenize(text)
        z = self.encode_with_transformer(tokens.to(self.device))
        return z

    def encode_with_transformer(self, text):
        x = self.model.token_embedding(text)  # [batch_size, n_ctx, d_model]
        x = x + self.model.positional_embedding
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.text_transformer_forward(x, attn_mask=self.model.attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.model.ln_final(x)
        return x

    def text_transformer_forward(self, x: torch.Tensor, attn_mask=None):
        for i, r in enumerate(self.model.transformer.resblocks):
            if i == len(self.model.transformer.resblocks) - self.layer_idx:
                break
            if (
                self.model.transformer.grad_checkpointing
                and not torch.jit.is_scripting()
            ):
                x = checkpoint(r, x, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x

    def encode(self, text):
        return self(text)
