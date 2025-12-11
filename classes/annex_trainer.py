# classes/annex_trainer.py

import winsound

from utils import *


class AnnexTrainer:
    def __init__(self, tokenizer, ids, load_checkpoint=False):
        self.model = TokenLSTM(vocab_size=tokenizer.vocab_size).to(DEVICE)
        if load_checkpoint:
            self.load_checkpoint()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LEARNING_RATE)
        self.train_data, self.val_data = make_splits(ids)

    def load_checkpoint(self):
        path = Path(SAMPLE_HISTORY_FILEPATH)
        state_dict = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(state_dict)

    def train(self, tokenizer):
        for step in range(MAX_ITERS):
            if step % EVAL_INTERVAL == 0:
                losses = estimate_loss(self.model, self.train_data, self.val_data, tokenizer.vocab_size)
                print(
                    f"Step {step}: "
                    f"train {losses['train']:.3f}, "
                    f"val {losses['val']:.3f}"
                )
                winsound.Beep(500, 250)

            x, y = get_batch("train", self.train_data, self.val_data)
            logits, _ = self.model(x)
            loss = F.cross_entropy(
                logits.view(-1, tokenizer.vocab_size),
                y.view(-1),
            )

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

        torch.save(self.model.state_dict(), MODEL_PATH)

    def generate(self, tokenizer: AnnexTokenizer, start_text: str = "Once upon a time") -> str:
        return generate(
            self.model,
            tokenizer=tokenizer,
            start_text=start_text
        )