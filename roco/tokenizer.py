from tokenizers import Tokenizer as HFTokenizer

class Tokenizer:
    def __init__(self, args):
        self.tokenizer = HFTokenizer.from_file('roco/tokenizer.json')

       
        self.max_length = args.max_length
        self.pad_id = self.tokenizer.token_to_id("<PAD>")
        self.bos_id = self.tokenizer.token_to_id("<BOS>")
        self.eos_id = self.tokenizer.token_to_id("<EOS>")

    def encode(self, text):
        # Add BOS and EOS tokens manually
        encoding = self.tokenizer.encode(
            self.bos_id + text + self.eos_id,
            add_special_tokens=False
        )
        ids = encoding

        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
        elif len(ids) < self.max_length:
            ids += [self.pad_id] * (self.max_length - len(ids))

        return ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)
