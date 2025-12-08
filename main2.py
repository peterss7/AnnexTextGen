from classes.trainer import Trainer
from classes.rosetta import Rosetta
from constants import *


def main():
    rosetta = Rosetta(CORPUS_FILEPATH)
    tokenizer = AnnexTokenizer(rosetta.vocab_size)
    trainer = Trainer(rosetta)

    trainer.train()

    print(trainer.generate(start_text="Once upon a time", max_new_tokens=300, temperature=0.6))


if __name__ == '__main__':
    main()