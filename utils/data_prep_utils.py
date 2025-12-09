from pathlib import Path

from classes import AnnexTokenizer
from constants import *


# ---------- Data prep ----------

def load_ids(tokenizer: AnnexTokenizer, corpus_path: str) -> torch.Tensor:
    text = Path(corpus_path).read_text(encoding="utf-8")
    ids = tokenizer.encode(text)
    return torch.tensor(ids, dtype=torch.long)


def make_splits(ids: torch.Tensor, train_ratio: float = 0.9):
    n = int(train_ratio * len(ids))
    train_data = ids[:n]
    val_data = ids[n:]
    return train_data, val_data


def get_batch(split: str, train_data, val_data):
    source = train_data if split == "train" else val_data
    ix = torch.randint(len(source) - BLOCK_SIZE - 1, (BATCH_SIZE,))
    x = torch.stack([source[i: i + BLOCK_SIZE] for i in ix])
    y = torch.stack([source[i + 1: i + 1 + BLOCK_SIZE] for i in ix])
    return x.to(DEVICE), y.to(DEVICE)

