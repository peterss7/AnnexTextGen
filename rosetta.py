from annex_gen import AnnexGen
from utils import Encoder as encoder
from utils import tutils
import torch

class Rosetta:
    def __init__(self, corpus_filepath: str):

        with open(corpus_filepath, "r", encoding="utf-8") as file:
            self.text = file.read()

        clean_text = tutils.clean_text(self.text)
        self.text = clean_text["clean_text"]
        self.chars = clean_text["chars"]
        self.vocab_size = clean_text["vocab_size"]
        self.str_to_index = {ch: i for i, ch in enumerate(self.chars)}
        self.index_to_str = {i: ch for ch, i in self.str_to_index.items()}
        self.data = encoder.encode(self.text, self.str_to_index)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AnnexGen(self.vocab_size).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)

