from types import MethodType
from typing import Any, Dict, List

import numpy as np
import torch
from torch import Tensor
from torch._C import device
from torch.nn import Module, Softmax
from transformers import BertConfig, BertModel, BertTokenizer
from transformers.modeling_bert import BertAttention, BertSelfAttention


class PatchedBertSelfAttention(Module):
    def __init__(
        self,
        model: BertSelfAttention,
        blind_spot: bool,
        drop_attn: bool,
        lmbda: float = 0.0,
    ):
        super().__init__()

        self.model = model

        self.blind_spot = blind_spot
        self.drop_attn = drop_attn
        self.lmbda = lmbda
        self._loss_value = torch.tensor(0.0)

        # Monkey patching the forward method.
        self.model.forward = MethodType(self.patch_forward, self.model)

    def forward(self, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        "This function is a thin wrapper for `BertSelfAttention`'s patched `forward`."

        return self.model(*args, **kwargs)

    @property
    def key(self):
        return self.model.key

    @property
    def query(self):
        return self.model.query

    @property
    def value(self):
        return self.model.value

    def loss(self) -> Tensor:
        """
        Additional losses that the patches generates.

        Returns
        -

        A scalar tensor that propagate loss values to the model.
        """

        if self.lmbda == 0.0:
            return torch.tensor(0.0)
        else:
            return self._loss_value

    def patch_forward(
        injected_self,
        self: BertSelfAttention,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        "This part of code is adapted from the `BertSelfAttention` class. It's the patched `forward` of `BertSelfAttention`"

        query = self.query(hidden_states)
        injected_self._loss_value = torch.tensor(0.0)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            kv_input = encoder_hidden_states
            attention_mask = encoder_attention_mask
        else:
            kv_input = hidden_states

        key = self.key(kv_input)
        value = self.value(kv_input)

        query = self.transpose_for_scores(query)
        key = self.transpose_for_scores(key)
        value = self.transpose_for_scores(value)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = query @ key.transpose(-1, -2)

        if injected_self.blind_spot:
            diag = torch.eye(attention_scores.shape[-1])
            for _ in range(attention_scores.ndim - diag.ndim):
                diag.unsqueeze_(0)

            assert attention_scores.shape[-1] == attention_scores.shape[-2]
            blind_spot_mask = 1.0 - diag
            assert blind_spot_mask.ndim == attention_scores.ndim
            attention_scores = attention_scores * blind_spot_mask

        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        if injected_self.lmbda != 0:
            diag = torch.eye(attention_scores.shape[-1])
            for _ in range(attention_scores.ndim - diag.ndim):
                diag.unsqueeze_(0)

            assert diag.ndim == attention_scores.ndim
            injected_self._loss_value = (
                injected_self._loss_value
                + (injected_self.lmbda * diag * attention_scores).sum()
            )

        if injected_self.drop_attn:
            # Dropout the tokens.
            attn_probs = self.dropout(attention_scores)
            attn_probs = Softmax(dim=-1)(attention_scores)
        else:
            # Normalize the attention scores to probabilities and dropout, as in the original transformer paper.
            attn_probs = Softmax(dim=-1)(attention_scores)
            attn_probs = self.dropout(attn_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attn_probs = attn_probs * head_mask

        # The values weighted by self attention.
        context = attn_probs @ value
        context = context.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context.shape[:-2] + (self.all_head_size,)
        context = context.view(*new_context_shape)

        return (context, attn_probs) if self.output_attentions else (context,)


class PatchedBert(Module):
    def __init__(
        self,
        model: BertModel,
        blind_spot: bool,
        drop_attn: bool,
        lmbda: float = 0.0,
    ):
        super().__init__()

        self.model = model
        for layer in self.model.encoder.layer:
            layer.attention.self = PatchedBertSelfAttention(
                layer.attention.self, blind_spot, drop_attn, lmbda
            )

    def forward(self, *args: List[Any], **kwargs: Dict[str, Any]) -> Any:
        return self.model(*args, **kwargs)


if __name__ == "__main__":
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert = PatchedBert(bert, True, True, 1)

    tnk = BertTokenizer.from_pretrained("bert-base-uncased")
    tok = tnk.tokenize("hello, world.")
    print(tok)
    enc = tnk.convert_tokens_to_ids(tok)
    print(enc)
    inp = tnk.build_inputs_with_special_tokens(enc)
    print(inp)

    tok1 = tnk.batch_encode_plus(
        ["hello, world.", "it's a beautiful sunday."],
        return_tensors="pt",
        pad_to_max_length=True,
    )
    print(tok1)

    out = bert(**tok1)
    print(out)
