# classes/token_lstm.py

from typing import Optional, Tuple

import torch.nn as nn
from torch import Tensor


class TokenLSTM(nn.Module):
    def __init__(
            self,
            vocab_size: int,
            embed_dim: int = 256,
            hidden_dim: int = 512,
            num_layers: int = 2,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(
            self,
            x: Tensor,
            hidden: Optional[Tuple[Tensor, Tensor]] = None,
    ) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        """
        x: (batch, seq_len)
        returns:
          logits: (batch, seq_len, vocab_size)
          hidden: (h_n, c_n)
        """
        x = self.embed(x)  # (batch, seq_len, embed_dim)
        out, hidden = self.lstm(x, hidden)  # (batch, seq_len, hidden_dim)
        logits = self.fc(out)  # (batch, seq_len, vocab_size)
        return logits, hidden
