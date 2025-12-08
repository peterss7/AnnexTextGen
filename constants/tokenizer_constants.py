ALLOWED_CHARS = string.ascii_letters + string.digits + string.punctuation + " \n\t"
TOKENS = {
    UNK_TOKEN: "[UNK]",
    PAD_TOKEN: "[PAD]",
    BOS_TOKEN: "[BOS]",
    EOS_TOKEN: "[EOS]",
}

TOKENS_ARR = [UNK_TOKEN, PAD_TOKEN, BOS_TOKEN, EOS_TOKEN]
SINGLE_TOKEN = f"{BOS_TOKEN} $A {EOS_TOKEN}"
PAIR_TOKEN = f"{BOS_TOKEN} $A {EOS_TOKEN} $B:1 {EOS_TOKEN}:1"