import os

import torch
from torch import nn
from transformers import BertModel

from .config import ModelConfig


class SpanAttention(nn.Module):
    """Span-level attention mechanism that attends to a fixed-size span of tokens for each query position.

    Args:
        hidden_size: Input feature dimension
        n_heads: Number of attention heads
        dim: Dimension per attention head
        span_size: Number of tokens in each span
        dropout: Dropout probability
    """
    def __init__(self, hidden_size: int, n_heads: int, dim: int, span_size: int, dropout: float = 0.1) -> None:
        super().__init__()
        self.span_size = span_size
        self.n_heads = n_heads
        self.dim = dim
        self.scaling = dim ** -0.5

        self.in_proj = nn.ModuleDict({
            key: nn.Linear(hidden_size, n_heads * dim)
            for key in ["query", "key", "value"]
        })
        self.out_proj = nn.Linear(n_heads * dim, hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize attention weights using Xavier uniform for projections."""
        for module in self.in_proj.values():
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.out_proj.bias is not None:
            nn.init.zeros_(self.out_proj.bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute span attention where each query attends to its corresponding span of keys/values.

        Args:
            q: Query tensor of shape (B, N, hidden_size)
            kv: Key/value tensor of shape (B, N*span_size, hidden_size)
            mask: Attention mask of shape (B, N*span_size)

        Returns:
            outputs: Attended output of shape (B, N, hidden_size)
            attn_wgts: Attention weights of shape (B, N, n_heads, span_size)
        """
        assert kv.size(1) == q.size(1) * self.span_size, "The length of kv must be span_size times the length of q"
        B, N, _ = q.shape

        q = self.in_proj["query"](q).view(B, N, self.n_heads, self.dim).unsqueeze(2).transpose(2, 3)  # B, N, H, 1, D
        k = self.in_proj["key"](kv).view(B, N, self.span_size, self.n_heads, self.dim).transpose(2, 3)  # B, N, H, S, D
        v = self.in_proj["value"](kv).view(B, N, self.span_size, self.n_heads, self.dim).transpose(2, 3)  # B, N, H, S, D

        attn_wgts = torch.matmul(q, k.transpose(-1, -2)) * self.scaling  # B, N, H, 1, S
        mask = mask.view(B, N, 1, 1, self.span_size)  # B, N, 1, 1, S
        attn_wgts = attn_wgts.masked_fill(mask == 0, float("-inf"))
        attn_wgts = nn.functional.softmax(attn_wgts, dim=-1)
        attn_wgts = self.dropout(attn_wgts)

        outputs = torch.matmul(attn_wgts, v).squeeze(3)  # B, N, H, D
        outputs = outputs.reshape(B, N, self.n_heads * self.dim)  # B, N, H*D
        outputs = self.out_proj(outputs)  # B, N, dim

        return outputs, attn_wgts.squeeze(-2)


class SpanOutput(nn.Module):
    """Transformer layer with span attention and feed-forward network for generating span representations.

    Args:
        hidden_size: Hidden dimension size
        n_heads: Number of attention heads
        dim: Dimension per head
        span_size: Size of each span
        dropout: Dropout probability
    """
    def __init__(self, hidden_size: int, n_heads: int, dim: int, span_size: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.span_size = span_size

        self.attention = SpanAttention(hidden_size, n_heads, dim, span_size, dropout)
        self.ln1 = nn.LayerNorm(hidden_size)
        self.ln2 = nn.LayerNorm(hidden_size)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_size, 4 * hidden_size),
            nn.GELU(),
            nn.Linear(4 * hidden_size, hidden_size),
            nn.Dropout(dropout)
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize layer normalization and feed-forward network weights."""
        # Initialize FFN layers
        for module in self.ffn:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor, kv: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply span attention followed by feed-forward network with residual connections.

        Args:
            x: Query input of shape (B, N, hidden_size)
            kv: Key/value input of shape (B, N*span_size, hidden_size)
            mask: Attention mask of shape (B, N*span_size)

        Returns:
            ffn_output: Final output of shape (B, N, hidden_size)
            attn_wgts: Attention weights
        """
        assert kv.size(1) == x.size(1) * self.span_size, "The length of kv must be span_size times the length of x"

        attn_output, attn_wgts = self.attention(x, kv, mask)
        span_output = self.ln1(x + attn_output)  # B, N, dim, N = x.size(1)
        ffn_output = self.ln2(span_output + self.ffn(span_output))

        return ffn_output, attn_wgts


class MSBert(nn.Module):
    """Multi-Scale BERT model that generates text representations at three granularities:
    CLS (coarse-grained), token (fine-grained), and span (mid-grained).

    Args:
        config: ModelConfig containing model hyperparameters
    """
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.device = config.device

        self.bert = BertModel.from_pretrained(config.pretrained_model)
        self.proj = nn.ModuleDict({
            key: nn.Linear(self.bert.config.hidden_size, config.dim)
            for key in ["cls", "tok", "span"]
        })
        self.span_output = SpanOutput(
            self.bert.config.hidden_size,
            config.n_heads,
            config.dim,
            config.span_size,
            config.dropout
        )

        self.init_weights()

    def init_weights(self) -> None:
        """Initialize projection layers."""
        # Initialize projection layers
        for module in self.proj.values():
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        include: list[str] = ["cls", "tok", "span"]
    ) -> torch.Tensor:
        """Encode input text into multi-scale representations.

        Args:
            input_ids: Token IDs of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            include: List of representation types to compute (choices: "cls", "tok", "span")

        Returns:
            Dictionary containing requested representations:
                - "cls": L2-normalized sentence embedding (B, D)
                - "tok": Dict with "repr" (B, T, D) and "mask" (B, T)
                - "span": Dict with "repr" (B, S, D), "attn_wgts", and "mask" (B, S)
        """
        repr = {}

        input_ids, attention_mask = input_ids.to(self.device), attention_mask.to(self.device)
        bert_outputs = self.bert(input_ids, attention_mask)[0]  # B, L, H

        if "cls" in include:
            cls_repr = self.proj["cls"](bert_outputs[:, 0])  # B, D
            cls_repr = torch.nn.functional.normalize(cls_repr, p=2, dim=-1)
            repr["cls"] = cls_repr

        # create masks: span_mask marks span boundary tokens, tok_mask marks regular tokens
        span_mask = ((torch.arange(input_ids.size(1), device=self.device) - 1) % self.config.span_size) == 0
        tok_mask = ~span_mask
        # exclude [CLS], [Q]/[D], and [SEP] tokens
        span_mask[:2] = 0
        span_mask[-1] = 0
        tok_mask[:2] = 0
        tok_mask[-1] = 0

        if "tok" in include:
            tok_repr = self.proj["tok"](bert_outputs[:,tok_mask])  # B, T, D
            tok_repr = tok_repr * attention_mask[:, tok_mask].unsqueeze(-1)
            tok_repr = torch.nn.functional.normalize(tok_repr, p=2, dim=-1)

            repr["tok"] = {
                "repr": tok_repr,
                "mask": attention_mask[:, tok_mask]
            }

        if "span" in include:
            mask = attention_mask[:, tok_mask].view(attention_mask.size(0), -1, self.config.span_size - 1).any(dim=-1)
            span_repr, attn_wgts = self.span_output(bert_outputs[:, span_mask], bert_outputs[:, 2:-1], attention_mask[:, 2:-1])
            span_repr = self.proj["span"](span_repr)  # B, S, D
            span_repr = span_repr * mask.unsqueeze(-1)
            span_repr = torch.nn.functional.normalize(span_repr, p=2, dim=-1)

            repr["span"] = {
                "repr": span_repr,
                "attn_wgts": attn_wgts,
                "mask": mask
            }

        return repr

    def save_pretrained(self, save_directory: str, save_config: bool = True) -> None:
        """Save model weights and configuration to directory.

        Args:
            save_directory: Directory path to save model
            save_config: Whether to save config.json
        """
        os.makedirs(save_directory, exist_ok=True)

        # save the model weights
        model_path = os.path.join(save_directory, "msbert.pth")
        torch.save(self.state_dict(), model_path)
        print(f"Model weights saved to {model_path}")

        # save the configuration (if required)
        if save_config and hasattr(self, "config"):
            import json

            config_path = os.path.join(save_directory, "config.json")
            with open(config_path, "w", encoding="utf-8") as f:
                json.dump(self.config.to_dict(), f, indent=2, ensure_ascii=False)
            print(f"Config saved to {config_path}")

        print(f"Model saved successfully to {save_directory}")

    @classmethod
    def from_pretrained(cls, model_path: str, config: ModelConfig | None = None) -> "MSBert":
        """Load a pretrained MSBert model from disk.

        Args:
            model_path: Directory containing saved model
            config: Optional ModelConfig; if None, loads from config.json

        Returns:
            Loaded MSBert model
        """
        # load the configuration of the model
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if os.path.exists(config_path):
                config = ModelConfig.from_json(config_path)
            else:
                raise ValueError(f"Config file not found at {config_path}")

        model = cls(config)

        # load the model weights
        checkpoint_path = os.path.join(model_path, "msbert.pth")
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Model weights not found at {model_path}")

        # load state dict
        state_dict = torch.load(checkpoint_path, map_location=config.device)

        # load weights into model
        model.load_state_dict(state_dict, strict=False)

        return model