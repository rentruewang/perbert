from .block import Embeddings, Encoder, Pooler
from .external import (
    BertConfig,
    BertForMaskedLM,
    BertForMultipleChoice,
    BertForNextSentencePrediction,
    BertForPreTraining,
    BertForPreTrainingOutput,
    BertForQuestionAnswering,
    BertForSequenceClassification,
    BertForTokenClassification,
    BertLMHeadModel,
    BertLMPredictionHead,
    BertModel,
    BertOnlyMLMHead,
    BertOnlyNSPHead,
    BertPredictionHeadTransform,
    BertPreTrainedModel,
    BertPreTrainingHeads,
)


class PredictionHeadTransform(BertPredictionHeadTransform):
    pass


class LMPredictionHead(BertLMPredictionHead):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.transform = PredictionHeadTransform(config)


class OnlyMLMHead(BertOnlyMLMHead):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.predictions = LMPredictionHead(config)


class OnlyNSPHead(BertOnlyNSPHead):
    pass


class PreTrainingHeads(BertPreTrainingHeads):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.predictions = LMPredictionHead(config)


class PreTrainedModel(BertPreTrainedModel):
    pass


class ForPreTrainingOutput(BertForPreTrainingOutput):
    pass


class Model(BertModel):
    def __init__(self, config: BertConfig, add_pooling_layer: bool = True):
        super().__init__(config, add_pooling_layer=add_pooling_layer)

        self.embeddings = Embeddings(config)
        self.encoder = Encoder(config)

        self.pooler = Pooler(config) if add_pooling_layer else None

        self.post_init()


class ForPreTraining(BertForPreTraining):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config)
        self.cls = PreTrainingHeads(config)

        self.post_init()


class LMHeadModel(BertLMHeadModel):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config, add_pooling_layer=False)
        self.cls = OnlyMLMHead(config)

        self.post_init()


class ForMaskedLM(BertForMaskedLM):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config, add_pooling_layer=False)
        self.cls = OnlyMLMHead(config)

        self.post_init()


class ForNextSentencePrediction(BertForNextSentencePrediction):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config)
        self.cls = OnlyNSPHead(config)

        self.post_init()


class ForSequenceClassification(BertForSequenceClassification):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config)

        self.post_init()


class ForMultipleChoice(BertForMultipleChoice):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config)

        self.post_init()


class ForTokenClassification(BertForTokenClassification):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config, add_pooling_layer=False)

        self.post_init()


class ForQuestionAnswering(BertForQuestionAnswering):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)

        self.bert = Model(config)

        self.post_init()
