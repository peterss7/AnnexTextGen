import torch

class EncoderUtils:

    def __init__(self, encoder):
        pass

    @staticmethod
    def encode(s: str, str_to_index: dict):
        return torch.tensor([str_to_index[ch] for ch in s], dtype=torch.long)

    @staticmethod
    def decode(indices, index_to_str: dict):
        return "".join(index_to_str[int(i)] for i in indices)