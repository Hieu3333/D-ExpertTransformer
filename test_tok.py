from tokenizers import Tokenizer
from modules.utils import load_all_keywords
# Load your trained tokenizer
tokenizer = Tokenizer.from_file("modules/custom_tokenizer.json")

# Test cases
examples = ["48-year-old black male. central serous retinopathy/ steroids-renal. re 20/30 le 20/200."]

for example in examples:
    encoding = tokenizer.encode(example)
    print(f"Input: {example}")
    print("Tokens:", encoding.tokens)
    print(f"Token IDs: {len(encoding.ids)}")
    print(f"Decode: {tokenizer.decode(encoding.ids)}")
    print()

keywords = load_all_keywords()
