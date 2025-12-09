# classes/annex_tokenizer.py

from pathlib import Path
from typing import List

from tokenizers import Tokenizer, pre_tokenizers, models, trainers
from tokenizers.processors import TemplateProcessing

from constants import *


class AnnexTokenizer:
    def __init__(
            self,
            corpus_path: str = CORPUS_PATH,
            tokenizer_path: str = TOKENIZER_PATH,
            vocab_size: int = VOCAB_SIZE,
    ) -> None:
        self.corpus_path = Path(corpus_path)
        self.tokenizer_path = Path(tokenizer_path)

        if not self.corpus_path.exists():
            raise FileNotFoundError(f"Corpus not found: {self.corpus_path}")

        if self.tokenizer_path.exists():
            self.tokenizer = Tokenizer.from_file(str(self.tokenizer_path))
        else:
            self.tokenizer = self._train_new_tokenizer(vocab_size)
            self.tokenizer.save(str(self.tokenizer_path))

    def _train_new_tokenizer(self, vocab_size: int) -> Tokenizer:
        # Base BPE model
        tokenizer = Tokenizer(models.BPE(unk_token=UNK))
        tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

        special_tokens = [PAD, UNK, BOS, EOS]

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=special_tokens,
        )

        tokenizer.train(
            files=[str(self.corpus_path)],
            trainer=trainer,
        )

        # Get IDs for BOS/EOS
        bos_id = tokenizer.token_to_id(BOS)
        eos_id = tokenizer.token_to_id(EOS)

        if bos_id is None or eos_id is None:
            raise RuntimeError("BOS/EOS ids are None; special_tokens not added correctly.")

        tokenizer.post_processor = TemplateProcessing(
            single=SINGLE_TOKENS,
            pair=PAIR_TOKENS,
            special_tokens=[
                (BOS, bos_id),
                (EOS, eos_id),
            ],
        )

        return tokenizer

    def encode(self, text: str) -> List[int]:
        # add_special_tokens=True is default when using post_processor
        return self.tokenizer.encode(text).ids

    def decode(self, ids: List[int]) -> str:
        return self.tokenizer.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.tokenizer.get_vocab_size()
