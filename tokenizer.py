import re
import json
from collections import Counter

class Tokenizer:
    def __init__(self, min_freq=2):
        self.min_freq = min_freq
        self.counter = Counter()
        self.important_words = ['re', 'le', 'od', 'os']

    def clean_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)  # Keep letters AND digits
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.lower().strip()

    def count_words(self, text):
        words = text.split()
        self.counter.update(words)

    def replace_rare(self, text):
        words = text.split()
        result = []
        for w in words:
            if self.counter[w] >= self.min_freq or w in self.important_words:
                result.append(w)
            else:
                result.append('<UNK>')
        return ' '.join(result)

# ================================
# Step 1: Load, Clean, Count Words
# ================================

tokenizer = Tokenizer(min_freq=2)
all_cleaned_text = []

# Store cleaned data temporarily before writing back
cleaned_data = {}

paths = [
    "data/DeepEyeNet_train.json",
    "data/DeepEyeNet_val.json",
    "data/DeepEyeNet_test.json"
]

cleaned_paths = [
    "data/cleaned_DeepEyeNet_train.json",
    "data/cleaned_DeepEyeNet_val.json",
    "data/cleaned_DeepEyeNet_test.json"
]

print("🔎 Starting cleaning and counting...")

for path in paths:
    print(f"Processing {path}...")
    with open(path, 'r') as f:
        anns = json.load(f)
    
    temp_clean = []  # Temporarily hold cleaned text for this file
    
    for item in anns:
        for meta in item.values():
            # Clean keywords
            keywords = meta.get('keywords', '')
            cleaned_keywords = ""
            if keywords:
                keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                cleaned_keywords_list = [tokenizer.clean_text(kw) for kw in keywords_list]
                cleaned_keywords = ', '.join(cleaned_keywords_list)
                all_cleaned_text.extend(cleaned_keywords_list)
            
            # Clean clinical_description
            description = meta.get('clinical_description', '')
            cleaned_description = ""
            if description:
                cleaned_description = tokenizer.clean_text(description)
                all_cleaned_text.append(cleaned_description)
            
            # Save cleaned (before replacing rare words) for now
            temp_clean.append({
                'keywords': cleaned_keywords,
                'clinical_description': cleaned_description
            })
    
    cleaned_data[path] = (anns, temp_clean)

# Count all words
for text in all_cleaned_text:
    tokenizer.count_words(text)

print(f"📊 Total vocab size before removing rare words: {len(tokenizer.counter)}")

# Optionally, save vocabulary to file
with open("data/vocab.json", "w") as vf:
    json.dump(dict(tokenizer.counter), vf, indent=2)
print(f"✅ Vocabulary saved to data/vocab.json")

# ================================
# Step 2: Replace Rare Words & Write to Cleaned Files
# ================================

print("\n📝 Replacing rare words and writing cleaned files...")

for orig_path, out_path in zip(paths, cleaned_paths):
    anns, temp_clean = cleaned_data[orig_path]
    
    print(f"Replacing rare words and writing to {out_path}...")

    idx = 0
    for item in anns:
        for meta in item.values():
            # Replace rare words in keywords
            cleaned_keywords = temp_clean[idx]['keywords']
            if cleaned_keywords:
                cleaned_kw_list = [tokenizer.replace_rare(kw) for kw in cleaned_keywords.split(', ')]
                meta['keywords'] = ', '.join(cleaned_kw_list)
            
            # Replace rare words in clinical_description
            cleaned_description = temp_clean[idx]['clinical_description']
            if cleaned_description:
                meta['clinical_description'] = tokenizer.replace_rare(cleaned_description)
            
            idx += 1
    
    # Write back cleaned data to new cleaned path
    with open(out_path, 'w') as f:
        json.dump(anns, f, indent=2)
    
    print(f"✅ Finished writing {out_path}.")

print("\n🎉 All files cleaned, rare words replaced, and saved to new cleaned files!")
