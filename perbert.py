import logging
from types import MethodType

import numpy as np
import torch
from rich.logging import RichHandler
from torch.nn import Softmax
from transformers import AlbertConfig, BertConfig
from transformers.modeling_albert import *
from transformers.modeling_bert import *

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler())

BERT_BASE_UNCASED = "bert-base-uncased"
ALBERT_BASE_V2 = "albert-base-v2"

NONE = "none"
MLMPAIR = "mlmpair"
SMALLSUBSET = "smallsubset"
BLINDSPOT = "blindspot"
SAVE10 = "save10"
FASTTERM = "fastterm"
RANDCROSS = "randcross"
MASKCROSS = "maskcross"

PATCHES = {
    NONE,
    MLMPAIR,
    SMALLSUBSET,
    BLINDSPOT,
    SAVE10,
    FASTTERM,
    RANDCROSS,
    MASKCROSS,
}


def PatchedBertSelfAttention(model, patches):
    if NONE in patches:
        logger.warning("Nothing is changed.")
        return model

    blindspot = BLINDSPOT in patches

    if blindspot:
        logger.warning("Blindspot is on.")

    randcross = RANDCROSS in patches

    if randcross:
        logger.warning("Random cross out is on.")

    def patch_forward(
        self,
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

        if randcross:
            length = attention_scores.shape[-1]
            device = attention_scores.device
            cross_mask = torch.zeros([length, length], device=device)
            for _ in range(attention_scores.ndim - cross_mask.ndim):
                cross_mask.unsqueeze_(0)

            cross_out = torch.rand([length], device=device)
            cross_mask[..., :, cross_out] -= 10000
            cross_mask[..., cross_out, :] -= 10000

        if attention_mask is None:
            attention_mask = cross_mask
        else:
            attention_mask = attention_mask + cross_mask

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


def Patched(model_type, model, patches):

    if NONE in patches:
        logger.warning("Nothing is changed.")
        return model

    if model_type == BERT_BASE_UNCASED:
        for layer in model.encoder.layer:
            layer.attention.self = PatchedBertSelfAttention(
                layer.attention.self, patches
            )

    if model_type == ALBERT_BASE_V2:
        for group in model.encoder.albert_layer_groups:
            for layer in group.albert_layers:
                layer.attention = PatchedBertSelfAttention(layer.attention, patches)
    return model


def PatchedForMaskedLM(model_type, model, patches):

    if NONE in patches:
        return model

    if model_type == BERT_BASE_UNCASED:
        model.bert = Patched(BERT_BASE_UNCASED, model.bert, patches)

    if model_type == ALBERT_BASE_V2:
        model.albert = Patched(ALBERT_BASE_V2, model.albert, patches)
        model.predictions.decoder = torch.nn.Linear(128, 30522)
    return model


def PatchedSequenceClassification(model_type, model, patches):

    if NONE in patches:
        logger.warning("Nothing is changed.")
        return model

    if model_type == BERT_BASE_UNCASED:

        if "random" in patches:
            logger.warning("random model used.")
            return BertForSequenceClassification(BertConfig())

        model.bert = Patched(BERT_BASE_UNCASED, model.bert, patches)
        return model

    if model_type == ALBERT_BASE_V2:
        if "random" in patches:
            return AlbertForSequenceClassification(AlbertConfig())

        model.albert = Patched(ALBERT_BASE_V2, model.albert, patches)
        return model

    logger.error("Fail %s", model_type)


if __name__ == "__main__":
    # bert = BertModel(BertConfig())
    # bert = Patched(BERT_BASE_UNCASED, bert, ["blindspot"])

    # albert = AlbertModel(AlbertConfig())
    # albert = Patched(ALBERT_BASE_V2, albert, ["blindspot"])

    PatchedForMaskedLM(ALBERT_BASE_V2, AlbertForMaskedLM(AlbertConfig()), [BLINDSPOT])
