from transformers import GPT2TokenizerFast

class Tokenizer:
    def __init__(self, args):
        self.tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

        # Add pad token if it's not already present
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})

        self.max_length = args.max_length
        self.pad_id = self.tokenizer.pad_token_id

    def encode(self, text):
        encoding = self.tokenizer.encode(text)
        ids = encoding

        if len(ids) > self.max_length:
            ids = ids[:self.max_length]
 
        elif len(ids) < self.max_length:
            ids += [self.pad_id] * (self.max_length - len(ids))

        return ids

    def decode(self, token_ids):
        return self.tokenizer.decode(token_ids, skip_special_tokens=True)



