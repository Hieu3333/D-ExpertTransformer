from modules.tokenizer import Tokenizer
from modules.utils import load_all_keywords
# Load your trained tokenizer
tokenizer = Tokenizer()
tokenizer.load_vocab('vocab.json')

# Test cases
examples = ["26 year old female amn macular neuroretinopathy"]

for example in examples:
    encoding = tokenizer.encode(example)
    print(f"Input: {example}")
    print("Tokens:", encoding)
    print(f"Token IDs: {len(encoding)}")
    print(f"Decode: {tokenizer.decode(encoding)}")
    print()
