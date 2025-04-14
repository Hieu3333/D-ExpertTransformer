from tokenizers import Tokenizer as HFTokenizer

class Tokenizer:
    def __init__(self, args=None):
        self.tokenizer = HFTokenizer.from_file('roco/tokenizer.json')

        # Get special token IDs from the tokenizer
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

        # Set max length
        self.max_length = args.max_length if args is not None else 50

    def encode(self, text):
        # Encode the text to get its token IDs
        encoding = self.tokenizer.encode(text)
        ids = encoding.ids

        # Truncate or pad the sequence
        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        elif len(ids) < self.max_length:
            ids += [self.pad_id] * (self.max_length - len(ids))

        return ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=False)


# tokenizer = Tokenizer()
# text = "A 3-year-old child with visual difficulties. Axial FLAIR image show a supra-sellar lesion"
# print(tokenizer.decode(tokenizer.encode(text)))