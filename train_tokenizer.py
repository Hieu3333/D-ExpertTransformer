import json
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.normalizers import Lowercase
from tokenizers.processors import TemplateProcessing
from pathlib import Path

# === Step 1: Load Dataset from Multiple Files ===
def load_texts(json_paths):
    texts = []
    for path in json_paths:
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            for _, value in item.items():
                if value.get('keywords'):
                    # Add the keywords with <SEP> inserted
                    keywords = value['keywords']
                    keywords_list = [kw.strip() for kw in keywords.split(',')]
                    sep_keywords = " <SEP> ".join(keywords_list)
                    texts.append(sep_keywords)
                if value.get('clinical-description'):
                    texts.append(value['clinical-description'])
    print(f"Total {len(texts)} text entries collected!")
    return texts

import json

def load_all_keywords():
    train_path = "data/DeepEyeNet_train.json"
    val_path = "data/DeepEyeNet_valid.json"
    test_path = "data/DeepEyeNet_test.json"
    all_keywords = set()
    json_paths = [train_path,val_path,test_path]
    for path in json_paths:
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            for _, value in item.items():
                if value.get('keywords'):
                    keywords = value['keywords']
                    # Split by comma and strip whitespace
                    keyword_list = [kw.strip() for kw in keywords.split(',')]
                    all_keywords.update(keyword_list)
    print(f"Total {len(all_keywords)} keywords collected!")
    return all_keywords


# === Step 2: Train Tokenizer ===
def train_tokenizer(texts, vocab_size=3500, save_path="custom_tokenizer.json"):
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # Normalization: lowercase
    tokenizer.normalizer = Lowercase()

    # Pre-tokenizer: whitespace splitting
    tokenizer.pre_tokenizer = Whitespace()

    # Special tokens
    special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<SEP>"]

    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=1,
        special_tokens=special_tokens
    )

    # Train tokenizer
    tokenizer.train_from_iterator(texts, trainer=trainer)

    # Post-processing to add BOS/EOS
    tokenizer.post_processor = TemplateProcessing(
        single="<BOS> $A <EOS>",
        pair="<BOS> $A <EOS> <BOS> $B <EOS>",
        special_tokens=[
            ("<BOS>", tokenizer.token_to_id("<BOS>")),
            ("<EOS>", tokenizer.token_to_id("<EOS>")),
        ],
    )

    tokenizer.save(save_path)
    print(f"Tokenizer saved to {save_path}")
    return tokenizer

# === Step 3: Encode Keywords with <SEP> ===
def encode_keywords(keywords, tokenizer):
    keyword_list = [kw.strip() for kw in keywords.split(',')]
    sep_joined = f" <SEP> ".join(keyword_list)
    encoded = tokenizer.encode(sep_joined)
    return encoded.ids


# === Step 3: Usage Example ===
def example_usage(tokenizer):
    sample = "presumed ocular histoplasmosis syndrome (pohs)"
    encoded_ids= encode_keywords(sample,tokenizer)
    print(f"\nSample text: {sample}")
    print("Token IDs:", encoded_ids)
    print("Decoded back:", tokenizer.decode(encoded_ids))


if __name__ == "__main__":
    # === Paths to JSON files ===
    train_path = "data/DeepEyeNet_train.json"
    val_path = "data/DeepEyeNet_valid.json"
    test_path = "data/DeepEyeNet_test.json"

    # === Load all texts ===
    all_texts = load_texts([train_path, val_path, test_path])

    # === Train tokenizer ===
    tokenizer = train_tokenizer(all_texts, vocab_size=5000, save_path="custom_tokenizer.json")

    
