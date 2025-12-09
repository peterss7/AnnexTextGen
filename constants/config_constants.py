# constants/config_constants.py
import torch


CORPUS_PATH = "./res/text/corpus.txt"
TOKENIZER_PATH = "./res/tokenizer.json"
MODEL_PATH = "./res/model.pt"
SAMPLE_HISTORY_FILEPATH = "./res/sample-history.json"

BLOCK_SIZE = 128  # number of tokens in context
BATCH_SIZE = 64
MAX_ITERS = 1000
EVAL_INTERVAL = 100
LEARNING_RATE = 1e-3

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PAD = "[PAD]"
UNK = "[UNK]"
BOS = "[BOS]"
EOS = "[EOS]"

SINGLE_TOKENS = "[BOS] $A [EOS]"
PAIR_TOKENS = "[BOS] $A [EOS] $B:1 [EOS]:1"

VOCAB_SIZE = 8000

MAX_NEW_TOKENS: int  = 400
TEMPERATURE = 0.7
TOP_K = 50