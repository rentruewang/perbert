from __future__ import annotations

from .external import BertAttention, BertConfig, BertSelfAttention, BertSelfOutput


class SelfAttention(BertSelfAttention):
    pass


class SelfOutput(BertSelfOutput):
    pass


class Attention(BertAttention):
    def __init__(
        self, config: BertConfig, position_embedding_type: str | None = None
    ) -> None:
        super().__init__(config, position_embedding_type=position_embedding_type)
        self.self = SelfAttention(config, position_embedding_type)
        self.output = SelfOutput(config)
