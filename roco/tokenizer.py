from transformers import Tokenizer as HFTokenizer

class Tokenizer:
    def __init__(self, args):
        self.tokenizer = HFTokenizer.from_pretrained('roco/tokenizer.json')

        # Add special tokens if not already present
        special_tokens_dict = {}
        if self.tokenizer.pad_token is None:
            special_tokens_dict["pad_token"] = "<PAD>"
        if self.tokenizer.bos_token is None:
            special_tokens_dict["bos_token"] = "<BOS>"
        if self.tokenizer.eos_token is None:
            special_tokens_dict["eos_token"] = "<EOS>"

        if special_tokens_dict:
            self.tokenizer.add_special_tokens(special_tokens_dict)

        self.max_length = args.max_length
        self.pad_id = self.tokenizer.pad_token_id
        self.bos_id = self.tokenizer.bos_token_id
        self.eos_id = self.tokenizer.eos_token_id

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
