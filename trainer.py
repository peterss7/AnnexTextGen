import random
import torch
import torch.nn.functional as F
from utils import Encoder as encoder

CONTEXT_LENGTH = 128



class Trainer:
    def __init__(self, rosetta, embed_dim=256, hidden_dim=512, num_layers=2):
        super().__init__()
        self.block_size = CONTEXT_LENGTH  # context length
        self.rosetta = rosetta
        data = self.rosetta.data
        n = int(0.9 * len(data))
        self.train_data = data[:n]
        self.val_data = data[n:]

    def get_batch(self, split, batch_size=64):
        source = self.train_data if split == "train" else self.val_data
        ix = torch.randint(len(source) - self.block_size - 1, (batch_size,))
        x = torch.stack([source[i: i + self.block_size] for i in ix])
        y = torch.stack([source[i + 1: i + 1 + self.block_size] for i in ix])

        return x, y  # shape: (batch_size, block_size)

    def estimate_loss(self, model):
        model.eval()
        out = {}
        for split in ["train", "val"]:
            losses = []
            for _ in range(10):  # a few batches to estimate
                x, y = self.get_batch(split, batch_size=64)
                x, y = x.to(self.rosetta.device), y.to(self.rosetta.device)
                with torch.no_grad():
                    logits, _ = model(x)
                    loss = F.cross_entropy(
                        logits.view(-1, self.rosetta.vocab_size),
                        y.view(-1)
                    )
                losses.append(loss.item())
            out[split] = sum(losses) / len(losses)
        model.train()
        return out

    def train(self, max_iters=1000, eval_interval=500, batch_size=64):

        for step in range(max_iters):
            if step % eval_interval == 0:
                losses = self.estimate_loss(self.rosetta.model)
                print(f"Step {step}: train {losses['train']:.3f}, val {losses['val']:.3f}")

            x, y = self.get_batch("train", batch_size=batch_size)
            x, y = x.to(self.rosetta.device), y.to(self.rosetta.device)

            logits, _ = self.rosetta.model(x)
            loss = F.cross_entropy(logits.view(-1, self.rosetta.vocab_size), y.view(-1))

            self.rosetta.optimizer.zero_grad()
            loss.backward()
            # optional but often helpful
            torch.nn.utils.clip_grad_norm_(self.rosetta.model.parameters(), max_norm=1.0)
            self.rosetta.optimizer.step()

    def generate(self, start_text="", max_new_tokens=500, temperature=1.0, top_k=50, block_size=128):
        self.rosetta.model.eval()
        with torch.no_grad():
            if start_text == "":
                # start from random character
                input_ids = torch.tensor([[random.randrange(self.rosetta.vocab_size)]], device=self.rosetta.device)
            else:
                input_ids = encoder.encode(start_text, self.rosetta.str_to_index).unsqueeze(0).to(self.rosetta.device)

            hidden = None
            generated = input_ids.clone()

            for _ in range(max_new_tokens):
                logits, hidden = self.rosetta.model(generated[:, -block_size:], hidden)
                # take logits of last time step
                last_logits = logits[:, -1, :].squeeze(0)
                next_id = self.sample_from_logits(last_logits, temperature=temperature, top_k=top_k)
                next_id_tensor = torch.tensor([[next_id]], device=self.rosetta.device)
                generated = torch.cat([generated, next_id_tensor], dim=1)

            return encoder.decode(generated[0].cpu(), self.rosetta.index_to_str)

    @staticmethod
    def sample_from_logits(logits, temperature=1.0, top_k=None):
        # logits: (vocab_size,)
        logits = logits / temperature

        if top_k is not None:
            values, indices = torch.topk(logits, top_k)
            mask = torch.full_like(logits, float("-inf"))
            mask[indices] = values
            logits = mask

        probs = F.softmax(logits, dim=-1)
        idx = torch.multinomial(probs, num_samples=1)
        return idx.item()
