# main.py
import json
import winsound

from classes.annex_trainer import AnnexTrainer
from utils import *

from classes.annex_tokenizer import AnnexTokenizer
from constants import *


def main():
    tokenizer = AnnexTokenizer()
    print("Vocab size:", tokenizer.vocab_size)

    ids = load_ids(tokenizer, CORPUS_PATH)
    print("Total tokens:", len(ids))



    annex_trainer = AnnexTrainer(tokenizer, ids)
    annex_trainer.train(tokenizer)
    sample = annex_trainer.generate(tokenizer, start_text="Where are we?")

    print("\n=== SAMPLE ===")
    print(sample)

    with open(SAMPLE_HISTORY_FILEPATH, "r") as file:
        history = json.load(file)

    history["sample_history"].append(sample)

    with open(SAMPLE_HISTORY_FILEPATH, "w", encoding="utf-8") as file:
        json.dump(history, file, indent=4)

    winsound.Beep(1000, 250)

if __name__ == "__main__":
    main()
