from types import MethodType

import numpy as np
import torch
from torch.nn import Module, Softmax
from torch.nn import functional as F
from transformers import BertModel, BertTokenizer
from transformers.modeling_bert import (
    BertForMaskedLM,
    BertForSequenceClassification,
    BertSelfAttention,
)


class DropSoftmax(Module):
    def __init__(self, p, dim) -> None:
        super().__init__()

        assert 0 <= p <= 1, p
        self._p_by_me = p
        self._dim_by_me = dim
        self._is_train_by_me = False

    def forward(self, *tensors):
        if len(tensors) == 1:
            return self.single_forward(*tensors)

        return [self.single_forward(t) for t in tensors]

    def single_forward(self, tensor):
        if not self._is_train_by_me:
            return F.softmax(tensor, dim=self._dim_by_me)

        select = torch.empty_like(tensor).uniform_(0, 1) < self._p_by_me
        remaining = torch.zeros_like(tensor)
        remaining[select] = -10000
        tensor = tensor + remaining
        out = F.softmax(tensor, dim=self._dim_by_me)
        return out

    def train(self, is_train=True):
        self._is_train_by_me = is_train
        return self

    def eval(self):
        self._is_train_by_me = False
        return self


def PatchedBertSelfAttention(
    model: BertSelfAttention,
    patches: list[str],
):
    if "none" in patches:
        return model

    blindspot = "blindspot" in patches
    dropsoft = "dropsoft" in patches

    if dropsoft:
        model.dropsoft = DropSoftmax(0.5, -1)

    def patch_forward(
        self: BertSelfAttention,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
    ):
        "This part of code is adapted from the `BertSelfAttention` class. It's the patched `forward` of `BertSelfAttention`"

        assert isinstance(self, BertSelfAttention), type(self)

        query = self.query(hidden_states)

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

        if blindspot:
            diag = torch.eye(attention_scores.shape[-1], device=attention_scores.device)
            for _ in range(attention_scores.ndim - diag.ndim):
                diag.unsqueeze_(0)

            assert attention_scores.shape[-1] == attention_scores.shape[-2]
            blind_spot_mask = -10000 * diag
            assert blind_spot_mask.ndim == attention_scores.ndim
            attention_scores = attention_scores

            if attention_mask is None:
                attention_mask = blind_spot_mask
            else:
                attention_mask = attention_mask + blind_spot_mask

        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities and dropout, as in the original transformer paper.
        if dropsoft:
            attn_probs = self.dropsoft(attention_scores)
        else:
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

    # Monkey patching the forward method.
    model.forward = MethodType(patch_forward, model)

    return model


def PatchedBert(
    model: BertModel,
    patches: list[str],
):
    if "none" in patches:
        return model

    assert isinstance(model, BertModel), type(model)
    for layer in model.encoder.layer:
        layer.attention.self = PatchedBertSelfAttention(layer.attention.self, version)
    return model


def PatchedBertForMaskedLM(
    model: BertForMaskedLM,
    patches: list[str],
):

    if "none" in patches:
        return model

    assert isinstance(model, BertForMaskedLM), type(model)
    model.bert = PatchedBert(model.bert, patches)
    return model


def PatchedBertForSequenceClassification(
    model: BertForSequenceClassification,
    patches: list[str],
):

    if "none" in patches:
        return model

    assert isinstance(model, BertForSequenceClassification), type(model)
    model.bert = PatchedBert(model.bert, patches)
    return model


if __name__ == "__main__":
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert = PatchedBert(bert, 3)

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
    bert.train()
    o = bert(**tok1)
    print(o)
