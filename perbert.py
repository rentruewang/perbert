from types import MethodType

import numpy as np
import torch
from torch.nn import Softmax
from transformers import BertModel, BertTokenizer
from transformers.modeling_bert import (
    BertForMaskedLM,
    BertSelfAttention,
    BertForSequenceClassification,
)


def PatchedBertSelfAttention(
    model: BertSelfAttention,
    blind_spot: bool,
    lmbda: float = 0.0,
):
    def lagrange(self):
        """
        Additional losses that the patches generates.

        Returns
        -

        A scalar tensor that propagate loss values to the model.
        """

        return self._lagrange

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
        self._lagrange = 0

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

        if blind_spot:
            diag = torch.eye(attention_scores.shape[-1], device=attention_scores.device)
            for _ in range(attention_scores.ndim - diag.ndim):
                diag.unsqueeze_(0)

            assert attention_scores.shape[-1] == attention_scores.shape[-2]
            blind_spot_mask = 1.0 - diag
            assert blind_spot_mask.ndim == attention_scores.ndim
            attention_scores = attention_scores * blind_spot_mask

        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

        # if lmbda != 0:
        #     diag = torch.eye(attention_scores.shape[-1], device=attention_scores.device)
        #     for _ in range(attention_scores.ndim - diag.ndim):
        #         diag.unsqueeze_(0)

        #     assert diag.ndim == attention_scores.ndim
        #     self._lagrange = self._lagrange + (lmbda * diag * attention_scores).sum()

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

    # Monkey patching the forward method.
    model._lagrange = 0
    model.forward = MethodType(patch_forward, model)
    model.lagrange = MethodType(lagrange, model)

    return model


def PatchedBert(
    model: BertModel,
    blind_spot: bool,
    orthogonal: float,
    lmbda: float = 0.0,
):
    assert isinstance(model, BertModel), type(model)
    for layer in model.encoder.layer:
        layer.attention.self = PatchedBertSelfAttention(
            layer.attention.self, blind_spot, lmbda
        )

    def lag_method(self):
        return sum(layer.attention.self.lagrange() for layer in self.encoder.layer)

    def ortho_method(self):
        weight = self.pooler.dense.weight
        matmul = weight @ weight.T
        return orthogonal * (torch.eye(matmul.shape[-1]) - matmul).pow(2).sum()

    model.lagrange = MethodType(lag_method, model)
    model.orthogonal = MethodType(ortho_method, model)

    return model


def PatchedBertForMaskedLM(
    model: BertForMaskedLM,
    blind_spot: bool,
    orthogonal: float,
    lmbda: float = 0.0,
):
    assert isinstance(model, BertForMaskedLM), type(model)
    model.bert = PatchedBert(model.bert, blind_spot, orthogonal, lmbda)
    return model


def PatchedBertForSequenceClassification(
    model: BertForSequenceClassification,
    blind_spot: bool,
    orthogonal: float,
    lmbda: float = 0.0,
):
    assert isinstance(model, BertForSequenceClassification), type(model)
    model.bert = PatchedBert(model.bert, blind_spot, orthogonal, lmbda)
    return model


if __name__ == "__main__":
    bert = BertModel.from_pretrained("bert-base-uncased")
    bert = PatchedBert(bert, True, 1, 1)
    print(hasattr(bert, "model"))

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
    print(len(out), list(o.shape for o in out))
    print(bert.lagrange())
    print(bert.orthogonal())

    torch.save(bert.state_dict(), open("model.pkl", "wb"))
    state_dict = torch.load(open("model.pkl", "rb"))
    # print(bert_rec)
    bert.load_state_dict(state_dict)
    out2 = bert(**tok1)
    print(out2)
