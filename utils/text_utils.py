import string

ALLOWED_CHARS = string.ascii_letters + string.digits + string.punctuation + " \n\t"

class TextUtils:
    @staticmethod
    def clean_text(text: str) -> dict:
        clean_text = "".join(char if char in ALLOWED_CHARS else " " for char in text)
        chars = sorted(list(set(clean_text)))
        vocab_size = len(chars)

        return { "clean_text": clean_text, "chars": chars, "vocab_size": vocab_size }

