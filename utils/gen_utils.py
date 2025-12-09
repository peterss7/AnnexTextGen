# utils/gen_utils.py

import random

import torch.nn.functional as F

from classes import AnnexTokenizer, TokenLSTM
from utils import get_batch
from constants import *


def estimate_loss(model, train_data, val_data, vocab_size: int):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = []
        for _ in range(10):  # a few batches
            x, y = get_batch(split, train_data, val_data)
            with torch.no_grad():
                logits, _ = model(x)
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size),
                    y.view(-1),
                )
            losses.append(loss.item())
        out[split] = sum(losses) / len(losses)
    model.train()
    return out


def sample_next_id(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None) -> int:
    # logits: (vocab_size,)
    logits = logits / temperature

    if top_k is not None:
        values, indices = torch.topk(logits, top_k)
        mask = torch.full_like(logits, float("-inf"))
        mask[indices] = values
        logits = mask

    probs = F.softmax(logits, dim=-1)
    idx = torch.multinomial(probs, num_samples=1)
    return int(idx.item())


def generate(
        model: TokenLSTM,
        tokenizer: AnnexTokenizer,
        start_text: str = "",
        max_new_tokens: int = MAX_NEW_TOKENS,
        temperature: float = TEMPERATURE,
        top_k: int | None = TOP_K,
) -> str:
    model.eval()
    with torch.no_grad():
        if start_text:
            input_ids = tokenizer.encode(start_text)
        else:
            # just start with BOS if defined, otherwise random token
            bos_id = tokenizer.tokenizer.token_to_id(BOS)
            if bos_id is None:
                input_ids = [random.randrange(tokenizer.vocab_size)]
            else:
                input_ids = [bos_id]

        x = torch.tensor([input_ids], dtype=torch.long, device=DEVICE)
        hidden = None

        for _ in range(int(max_new_tokens)):
            x_cond = x[:, -BLOCK_SIZE:]
            logits, hidden = model(x_cond, hidden)
            last_logits = logits[:, -1, :].squeeze(0)
            next_id = sample_next_id(last_logits, temperature=temperature, top_k=top_k)
            next_id_tensor = torch.tensor([[next_id]], device=DEVICE)
            x = torch.cat([x, next_id_tensor], dim=1)

        ids = x[0].tolist()
        return tokenizer.decode(ids)
