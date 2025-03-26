from collections import Counter
from tokenizers import Tokenizer
import json

def analyze_length_distribution(tokenizer):
    train_path = "data/cleaned_DeepEyeNet_train.json"
    val_path = "data/cleaned_DeepEyeNet_val.json"
    test_path = "data/cleaned_DeepEyeNet_test.json"
    
    json_paths = [train_path, val_path, test_path]
    length_counts = Counter()  # Dictionary to store frequency of tokenized lengths

    for path in json_paths:
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            for _, value in item.items():
                if value.get('clinical-description'):
                    text = value['clinical-description']
                    encoded = tokenizer.encode(text)
                    token_length = len(encoded.ids)
                    length_counts[token_length] += 1  # Count occurrences

    # Print statistics
    total_entries = sum(length_counts.values())
    avg_entry_length = sum(k * v for k, v in length_counts.items()) / total_entries if total_entries > 0 else 0
    max_entry_length = max(length_counts.keys(), default=0)

    print(f"\nTotal Clinical Descriptions: {total_entries}")
    print(f"Average Token Length Per Entry: {avg_entry_length:.2f}")
    print(f"Max Token Length Among All Entries: {max_entry_length}")
    
    print("\nToken Length Distribution (Length: Count):")
    for length, count in sorted(length_counts.items()):
        print(f"{length}: {count}")

# Load the trained tokenizer
tokenizer = Tokenizer.from_file("custom_tokenizer.json")

# Analyze clinical descriptions
analyze_length_distribution(tokenizer)
