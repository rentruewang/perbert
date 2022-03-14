from torch.nn import ModuleList

from .attention import Attention
from .external import (
    BertConfig,
    BertEmbeddings,
    BertEncoder,
    BertIntermediate,
    BertLayer,
    BertOutput,
    BertPooler,
)


class Embeddings(BertEmbeddings):
    pass


class Intermediate(BertIntermediate):
    pass


class Output(BertOutput):
    pass


class Layer(BertLayer):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.attention = Attention(config)
        self.intermediate = Intermediate(config)
        self.output = Output(config)

        if self.add_cross_attention:
            self.crossattention = Attention(config, "absolute")


class Encoder(BertEncoder):
    def __init__(self, config: BertConfig) -> None:
        super().__init__(config)
        self.layer = ModuleList(
            (Layer(config) for _ in range(config.num_hidden_layers))
        )


class Pooler(BertPooler):
    pass
