from tokenizers import Tokenizer as HFTokenizer

class Tokenizer:
    def __init__(self, args):
        self.tokenizer = HFTokenizer.from_file('roco/tokenizer.json')

       
        self.max_length = args.max_length
        self.pad_id = self.tokenizer.pad_token
        self.bos_id = self.tokenizer.bos_token
        self.eos_id = self.tokenizer.eos_token

    def encode(self, text):
        # Add BOS and EOS tokens manually
        encoding = self.tokenizer.encode(
            self.tokenizer.bos_token + text + self.tokenizer.eos_token,
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
