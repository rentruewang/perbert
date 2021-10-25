import logging
from types import MethodType

import numpy as np
import torch
from rich.logging import RichHandler
from torch import Tensor
from torch.nn import Module, Softmax, Embedding
from transformers import BertConfig
from transformers.modeling_albert import *
from transformers.modeling_bert import *
from transformers.tokenization_bert import BertTokenizer

logger = logging.getLogger(__name__)
logger.addHandler(RichHandler(logging.WARNING))

BERT_BASE_UNCASED = "bert-base-uncased"
# ALBERT_BASE_V2 = "albert-base-v2"

NONE = "none"
MLMPAIR = "mlmpair"
SMALLSUBSET = "smallsubset"
MASKSLASH = "maskslash"
MASKDOT = "maskdot"
ANDDECAY = "anddecay"
SAVE10 = "save10"
FASTTERM = "fastterm"
RANDCROSS = "randcross"
MASKCROSS = "maskcross"
EMBEDDING = "embedding"
FIRSTLAYER = "firstlayer"
EARLYFOCUS = "earlyfocus"

CROSS_ATTN_PROB = 0.15

PATCHES = {
    NONE,
    MLMPAIR,
    SMALLSUBSET,
    MASKSLASH,
    MASKDOT,
    ANDDECAY,
    SAVE10,
    FASTTERM,
    RANDCROSS,
    MASKCROSS,
    EMBEDDING,
    FIRSTLAYER,
    EARLYFOCUS,
}


class CrossDropout(Module):
    def __init__(self, mask_cross: bool, mask_dot: bool, and_decay: bool) -> None:
        super().__init__()
        self.mask_cross = mask_cross
        self.mask_dot = mask_dot
        if and_decay:
            if not self.mask_cross:
                raise ValueError("this is stupid.")

            self.rate = 0
        else:
            self.rate = 10000

        logger.warning("Initial rate: %f", self.rate)
        logger.warning(
            "Dot: %s, Cross: %s, Decay: %s", self.mask_dot, self.mask_cross, and_decay
        )

    def forward(self, drop: Tensor) -> Tensor:
        assert drop.ndim == 2, drop.shape

        mask = torch.zeros(
            [
                drop.shape[0],
                1,
                drop.shape[1],
                drop.shape[1],
            ],
            device=drop.device,
        )

        if not self.training:
            return mask

        if self.mask_cross:
            for (m, d) in zip(mask, drop):
                m[:, :, d] -= self.rate
                m[:, d, :] -= self.rate
            self.rate += 1 / 1000

        if self.mask_dot:
            for (m, d) in zip(mask, drop):
                m[:, d, d] -= 10000.0

        return mask


def patch_bertenc_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    masked_lm_labels=None,
):
    assert isinstance(self, BertEncoder), type(self)
    all_hidden_states = ()
    all_attentions = ()
    for i, layer_module in enumerate(self.layer):
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        layer_outputs = layer_module(
            hidden_states,
            attention_mask,
            head_mask[i],
            encoder_hidden_states,
            encoder_attention_mask,
            masked_lm_labels=masked_lm_labels,
        )
        hidden_states = layer_outputs[0]

        if self.output_attentions:
            all_attentions = all_attentions + (layer_outputs[1],)

    # Add last layer
    if self.output_hidden_states:
        all_hidden_states = all_hidden_states + (hidden_states,)

    outputs = (hidden_states,)
    if self.output_hidden_states:
        outputs = outputs + (all_hidden_states,)
    if self.output_attentions:
        outputs = outputs + (all_attentions,)
    return outputs  # last-layer hidden state, (all hidden states), (all attentions)


def patch_bertenclay_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    masked_lm_labels=None,
):
    assert isinstance(self, BertLayer), type(self)
    self_attention_outputs = self.attention(
        hidden_states, attention_mask, head_mask, masked_lm_labels=masked_lm_labels
    )
    attention_output = self_attention_outputs[0]
    outputs = self_attention_outputs[
        1:
    ]  # add self attentions if we output attention weights

    if self.is_decoder and encoder_hidden_states is not None:
        cross_attention_outputs = self.crossattention(
            attention_output,
            attention_mask,
            head_mask,
            encoder_hidden_states,
            encoder_attention_mask,
        )
        attention_output = cross_attention_outputs[0]
        outputs = (
            outputs + cross_attention_outputs[1:]
        )  # add cross attentions if we output attention weights

    intermediate_output = self.intermediate(attention_output)
    layer_output = self.output(intermediate_output, attention_output)
    outputs = (layer_output,) + outputs
    return outputs


def patch_bertattn_forward(
    self,
    hidden_states,
    attention_mask=None,
    head_mask=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    masked_lm_labels=None,
):
    assert isinstance(self, BertAttention), type(self)
    self_outputs = self.self(
        hidden_states,
        attention_mask,
        head_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        masked_lm_labels=masked_lm_labels,
    )
    attention_output = self.output(self_outputs[0], hidden_states)
    outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them
    return outputs


def patched_bertmlm_forward(
    self,
    input_ids=None,
    attention_mask=None,
    token_type_ids=None,
    position_ids=None,
    head_mask=None,
    inputs_embeds=None,
    masked_lm_labels=None,
    encoder_hidden_states=None,
    encoder_attention_mask=None,
    lm_labels=None,
):
    assert isinstance(self, BertForMaskedLM), type(self)
    outputs = self.bert(
        input_ids,
        attention_mask=attention_mask,
        token_type_ids=token_type_ids,
        position_ids=position_ids,
        head_mask=head_mask,
        inputs_embeds=inputs_embeds,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        masked_lm_labels=masked_lm_labels,
    )

    sequence_output = outputs[0]
    prediction_scores = self.cls(sequence_output)

    outputs = (prediction_scores,) + outputs[
        2:
    ]  # Add hidden states and attention if they are here

    # Although this may seem awkward, BertForMaskedLM supports two scenarios:
    # 1. If a tensor that contains the indices of masked labels is provided,
    #    the cross-entropy is the MLM cross-entropy that measures the likelihood
    #    of predictions for masked words.
    # 2. If `lm_labels` is provided we are in a causal scenario where we
    #    try to predict the next token for each input in the decoder.
    if masked_lm_labels is not None:
        loss_fct = CrossEntropyLoss()  # -100 index = padding token
        masked_lm_loss = loss_fct(
            prediction_scores.view(-1, self.config.vocab_size),
            masked_lm_labels.view(-1),
        )
        outputs = (masked_lm_loss,) + outputs

    assert lm_labels is None

    return outputs  # (masked_lm_loss), (ltr_lm_loss), prediction_scores, (hidden_states), (attentions)


def PatchedBertSelfAttention(model, patches, layer_num):
    if NONE in patches:
        logger.warning("Nothing is changed.")
        return model

    maskslash = MASKSLASH in patches
    maskdot = MASKDOT in patches
    anddecay = ANDDECAY in patches
    randcross = RANDCROSS in patches
    maskcross = MASKCROSS in patches
    firstlayer = FIRSTLAYER in patches

    if firstlayer:
        if layer_num == 0:
            logger.warning("First layer %d on.", layer_num)
        else:
            logger.warning("Layer %d not first layer. Features turned off.", layer_num)
            maskslash = maskdot = anddecay = randcross = maskcross = False

    if maskslash:
        logger.warning("Mask slash is on.")

    if randcross:
        logger.warning("Random cross out is on.")

    if maskcross:
        logger.warning("Mask cross is enabled.")

    if maskdot:
        logger.warning("Mask dot is on.")

    if anddecay:
        logger.warning("And decay is on.")

    def patch_bertenclayattn_forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        masked_lm_labels=None,
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

        if maskslash:
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

            cross_out = torch.rand([1, length], device=device)
            cross_out = cross_out < CROSS_ATTN_PROB
            cross_mask = self.cross_drop(cross_out)

            assert cross_mask.ndim == attention_scores.ndim, [
                cross_mask.shape,
                attention_scores.shape,
            ]

            if attention_mask is None:
                attention_mask = cross_mask
            else:
                attention_mask = attention_mask + cross_mask

        if maskcross or maskdot:
            assert masked_lm_labels is not None
            length = attention_scores.shape[-1]
            device = attention_scores.device
            cross_mask = self.cross_drop(masked_lm_labels == -100)
            assert cross_mask.ndim == attention_scores.ndim, [
                cross_mask.shape,
                attention_scores.shape,
            ]

            if attention_mask is None:
                attention_mask = cross_mask
            else:
                attention_mask = attention_mask + cross_mask

        attention_scores = attention_scores / np.sqrt(self.attention_head_size)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask

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

    model.cross_drop = CrossDropout(maskcross, maskdot, anddecay)
    model.forward = MethodType(patch_bertenclayattn_forward, model)

    return model


def PatchedBert(model_type, model, patches):

    if NONE in patches:
        logger.warning("Nothing is changed.")
        return model

    if model_type == BERT_BASE_UNCASED:
        model.encoder.forward = MethodType(patch_bertenc_forward, model.encoder)
        for (idx, layer) in enumerate(model.encoder.layer):
            logger.warning("Substituting Bert layer %d", idx)
            layer.forward = MethodType(patch_bertenclay_forward, layer)
            layer.attention.forward = MethodType(
                patch_bertattn_forward, layer.attention
            )
            layer.attention.self = PatchedBertSelfAttention(
                layer.attention.self, patches, idx
            )

        yes_embedding = EMBEDDING in patches
        if yes_embedding:
            logger.warning("Only embedding layer is used.")

        def patch_bert_forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            encoder_hidden_states=None,
            encoder_attention_mask=None,
            masked_lm_labels=None,
        ):
            assert isinstance(self, BertModel), type(self)

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError(
                    "You cannot specify both input_ids and inputs_embeds at the same time"
                )
            elif input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError(
                    "You have to specify either input_ids or inputs_embeds"
                )

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(
                    input_shape, dtype=torch.long, device=device
                )

            # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
            # ourselves in which case we just need to make it broadcastable to all heads.
            if attention_mask.dim() == 3:
                extended_attention_mask = attention_mask[:, None, :, :]
            elif attention_mask.dim() == 2:
                # Provided a padding mask of dimensions [batch_size, seq_length]
                # - if the model is a decoder, apply a causal mask in addition to the padding mask
                # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
                if self.config.is_decoder:
                    batch_size, seq_length = input_shape
                    seq_ids = torch.arange(seq_length, device=device)
                    causal_mask = (
                        seq_ids[None, None, :].repeat(batch_size, seq_length, 1)
                        <= seq_ids[None, :, None]
                    )
                    causal_mask = causal_mask.to(
                        attention_mask.dtype
                    )  # causal and attention masks must have same type with pytorch version < 1.3
                    extended_attention_mask = (
                        causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
                    )
                else:
                    extended_attention_mask = attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                        input_shape, attention_mask.shape
                    )
                )

            # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
            # masked positions, this operation will create a tensor which is 0.0 for
            # positions we want to attend and -10000.0 for masked positions.
            # Since we are adding it to the raw scores before the softmax, this is
            # effectively the same as removing these entirely.
            extended_attention_mask = extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

            # If a 2D ou 3D attention mask is provided for the cross-attention
            # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder and encoder_hidden_states is not None:
                (
                    encoder_batch_size,
                    encoder_sequence_length,
                    _,
                ) = encoder_hidden_states.size()
                encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
                if encoder_attention_mask is None:
                    encoder_attention_mask = torch.ones(
                        encoder_hidden_shape, device=device
                    )

                if encoder_attention_mask.dim() == 3:
                    encoder_extended_attention_mask = encoder_attention_mask[
                        :, None, :, :
                    ]
                elif encoder_attention_mask.dim() == 2:
                    encoder_extended_attention_mask = encoder_attention_mask[
                        :, None, None, :
                    ]
                else:
                    raise ValueError(
                        "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                            encoder_hidden_shape, encoder_attention_mask.shape
                        )
                    )

                encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                    dtype=next(self.parameters()).dtype
                )  # fp16 compatibility
                encoder_extended_attention_mask = (
                    1.0 - encoder_extended_attention_mask
                ) * -10000.0
            else:
                encoder_extended_attention_mask = None

            # Prepare head mask if needed
            # 1.0 in head_mask indicate we keep the head
            # attention_probs has shape bsz x n_heads x N x N
            # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
            # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = (
                        head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    )
                    head_mask = head_mask.expand(
                        self.config.num_hidden_layers, -1, -1, -1, -1
                    )
                elif head_mask.dim() == 2:
                    head_mask = (
                        head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                    )  # We can specify head_mask for each layer
                head_mask = head_mask.to(
                    dtype=next(self.parameters()).dtype
                )  # switch to fload if need + fp16 compatibility
            else:
                head_mask = [None] * self.config.num_hidden_layers

            embedding_output = self.embeddings(
                input_ids=input_ids,
                position_ids=position_ids,
                token_type_ids=token_type_ids,
                inputs_embeds=inputs_embeds,
            )

            if yes_embedding:
                sequence_output = embedding_output
                encoder_outputs = tuple([sequence_output])
            else:
                encoder_outputs = self.encoder(
                    embedding_output,
                    attention_mask=extended_attention_mask,
                    head_mask=head_mask,
                    encoder_hidden_states=encoder_hidden_states,
                    encoder_attention_mask=encoder_extended_attention_mask,
                    masked_lm_labels=masked_lm_labels,
                )
                sequence_output = encoder_outputs[0]
            pooled_output = self.pooler(sequence_output)

            outputs = (sequence_output, pooled_output,) + encoder_outputs[
                1:
            ]  # add hidden_states and attentions if they are here
            return (
                outputs  # sequence_output, pooled_output, (hidden_states), (attentions)
            )

        model.forward = MethodType(patch_bert_forward, model)

    # if model_type == ALBERT_BASE_V2:
    #     for group in model.encoder.albert_layer_groups:
    #         for layer in group.albert_layers:
    #             layer.attention = PatchedBertSelfAttention(layer.attention, patches)
    return model


def PatchedForMaskedLM(model_type, model, patches):

    if NONE in patches:
        return model

    if model_type == BERT_BASE_UNCASED:
        model.bert = PatchedBert(BERT_BASE_UNCASED, model.bert, patches)
        model.forward = MethodType(patched_bertmlm_forward, model)

    # if model_type == ALBERT_BASE_V2:
    #     model.albert = Patched(ALBERT_BASE_V2, model.albert, patches)
    #     model.predictions.decoder = torch.nn.Linear(128, 30522)
    return model


def PatchedSequenceClassification(model_type, model, patches):

    if NONE in patches:
        logger.warning("Nothing is changed.")
        return model

    if model_type == BERT_BASE_UNCASED:

        if "random" in patches:
            logger.warning("random model used.")
            return BertForSequenceClassification(BertConfig())

        model.bert = PatchedBert(BERT_BASE_UNCASED, model.bert, patches)
        return model

    # if model_type == ALBERT_BASE_V2:
    #     if "random" in patches:
    #         return AlbertForSequenceClassification(AlbertConfig())

    #     model.albert = Patched(ALBERT_BASE_V2, model.albert, patches)
    #     return model

    logger.error("Fail %s", model_type)


if __name__ == "__main__":
    bert = BertModel(BertConfig())
    bert = PatchedBert(BERT_BASE_UNCASED, bert, [MASKSLASH, RANDCROSS])

    tokenizer = BertTokenizer.from_pretrained(BERT_BASE_UNCASED)
    texts = ["hello world", "goodbye world"]
    inputs = tokenizer.batch_encode_plus(texts, max_length=512, return_tensors="pt")
    out = bert(**inputs)
    print(out)

    # albert = AlbertModel(AlbertConfig())
    # albert = Patched(ALBERT_BASE_V2, albert, [MASKSLASH])

    # PatchedForMaskedLM(ALBERT_BASE_V2, AlbertForMaskedLM(AlbertConfig()), [MASKSLASH])
