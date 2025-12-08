import string
from constants import *


class AnnexTokenizer:
    def __init__(self, vocab_size: number):
        # self.tokenizer = Tokenizer(models.BPE(unk_token=UNK_TOKEN))
        self.tokenizer = Tokenizer.from_file(TOKENIZER_FILEPATH)
        self.pre_tokenizer = pre_tokenizers.Whitespace()
        self.trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS
        )
        self.tokenizer.post_processor = TemplateProcessing(
            single=SINGLE_TOKEN,
            pair=PAIR_TOKEN,
            special_tokens=[
                (TOKENS.BOS_TOKEN, tokenizer.token_to_id(TOKENS.BOS_TOKEN)),
                (TOKENS.EOS_TOKEN, tokenizer.token_to_id(TOKENS.EOS_TOKEN)),
            ],
        )
        self.vocab_size = self.tokenizer.get_vocab_size()


    def train_tokenizer(self, text: str):
        self.tokenizer.train(
            text,
            trainer=self.trainer
        )
        self.tokenizer.save(TOKENIZER_FILENAME)


    def encode(self):
        return self.tokenizer.encode(text).ids

    def decode(self, ids):
        return self.tokenizer.decode(ids)
