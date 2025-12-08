import torch.nn as nn
import torch.nn.functional as F

class AnnexGen(nn.Module):
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch, seq_len)
        x = self.embed(x)  # -> (batch, seq_len, embed_dim)
        out, hidden = self.lstm(x, hidden)  # out: (batch, seq_len, hidden_dim)
        logits = self.fc(out)  # -> (batch, seq_len, vocab_size)
        return logits, hidden