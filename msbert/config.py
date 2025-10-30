from dataclasses import dataclass, asdict

import torch


@dataclass
class TokenizerConfig:
    pretrained_model: str = "bert-base-uncased"

    span_size: int = 16

    qry_maxlen: int = 64
    qry_token: str = "[Q]"
    qry_token_id: str = "[unused0]"

    doc_maxlen: int = 512
    doc_token: str = "[D]"
    doc_token_id: str = "[unused1]"

    span_token: str = "[S]"
    span_token_id: str = "[unused2]"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelConfig:
    pretrained_model: str = "bert-base-uncased"

    dim: int = 128
    n_heads: int = 8
    span_size: int = 16
    dropout: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TrainConfig:
    span_size: int = 16
    qry_maxlen: int = 64
    doc_maxlen: int = 512

    pretrained_model: str = "bert-base-uncased"

    dim: int = 128
    n_heads: int = 8
    dropout: float = 0.1

    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    bsize: int = 16
    n_negative: int = 3
    maxsteps: int = 30_000
    warmup: int = 3_000
    lr: float = 5e-6
    neg_sampling: bool = False
    temperature: float = 0.1

    def __post_init__(self):
        self.model_config = ModelConfig(
            pretrained_model=self.pretrained_model,
            dim=self.dim,
            n_heads=self.n_heads,
            span_size=self.span_size,
            dropout=self.dropout,
            device=self.device
        )

        self.tok_config = TokenizerConfig(
            pretrained_model=self.pretrained_model,
            span_size=self.span_size - 1,
            qry_maxlen=self.qry_maxlen,
            doc_maxlen=self.doc_maxlen,
        )

    def to_dict(self) -> dict:
        return asdict(self)