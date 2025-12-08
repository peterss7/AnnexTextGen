from trainer import Trainer
from rosetta import Rosetta

CORPUS_FILEPATH = "./res/text/corpus.txt"

def main():
    rosetta = Rosetta(CORPUS_FILEPATH)
    trainer = Trainer(rosetta)

    trainer.train()

    print(trainer.generate(start_text="Once upon a time", max_new_tokens=300, temperature=0.6))


if __name__ == '__main__':
    main()