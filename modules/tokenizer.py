import re
import json
from collections import Counter

class Tokenizer:
    def __init__(self,args=None):
        self.counter = Counter()
        self.special_tokens = ['<PAD>', '<UNK>', '<SEP>', '<BOS>', '<EOS>','<MASK>']
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        self.max_length = args.max_length if args is not None else 0

    def clean_text(self, text):
        """Clean text by removing unwanted characters and normalizing spaces."""
        # Keep letters, digits, spaces, apostrophes, slashes, hyphens, dots, and parentheses
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # Remove hyphens only between digits and words (30-year-old → 30 year old)
        # text = re.sub(r'(?<=\d)-(?=\w)|(?<=\w)-(?=\d)', ' ', text)
        
        # Replace multiple hyphens with single space (for remaining hyphens)
        # text = re.sub(r'-+', ' ', text)
        
        # Normalize spaces 
        text = re.sub(r'\s+', ' ', text).strip()
        # text = re.sub(r'\.',' . ',text)
        # text = re.sub(r',',' , ',text)
        return text.lower()

    def collect_texts(self, filepaths):
        """Load JSON files and collect all clinical-descriptions and keywords."""
        all_cleaned_text = []
        data_store = {}  # Store cleaned texts for later use
        
        for path in filepaths:
            with open(path, 'r') as f:
                anns = json.load(f)
            
            cleaned_entries = []
            for item in anns:
                for meta in item.values():
                    # Process keywords
                    keywords = meta.get('keywords', '')
                    cleaned_keywords = ""
                    if keywords:
                        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
                        cleaned_keywords_list = [self.clean_text(kw) for kw in keywords_list]
                        cleaned_keywords = ', '.join(cleaned_keywords_list)
                        all_cleaned_text.extend(cleaned_keywords_list)

                    # Process clinical-description
                    description = meta.get('clinical-description', '')
                    cleaned_description = ""
                    if description:
                        cleaned_description = self.clean_text(description)
                        all_cleaned_text.append(cleaned_description)

                    cleaned_entries.append({
                        'keywords': cleaned_keywords,
                        'clinical-description': cleaned_description,
                        'original': description
                    })

            data_store[path] = (anns, cleaned_entries)
        
        return all_cleaned_text, data_store

    def build_vocab(self, all_texts):
        """Count words and build vocabulary, replacing words that appear once with <UNK>."""
        # Count all word occurrences
        for text in all_texts:
            self.counter.update(text.split())

        # Replace rare words (words appearing only once) with <UNK>
        for word, freq in self.counter.items():
            if freq > 1:  # Keep words appearing more than once
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
            else:
                print(word)


    def replace_rare(self, text):
        """Replace words that appear only once with <UNK>."""
        words = text.split()
        return ' '.join(w if w in self.word2idx else '<UNK>' for w in words)

    def save_vocab(self, filepath):
        """Save vocabulary to a JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.word2idx, f, indent=2)

    def load_vocab(self, filepath):
        """Load vocabulary from a JSON file."""
        with open(filepath, 'r') as f:
            self.word2idx = json.load(f)
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def encode(self, text):
        """
        Convert a text into a list of token indices.
        Adds <BOS> at the start and <EOS> at the end.
        Replaces unknown words with <UNK>.
        """
        tokens = text.split()
        token_ids = [self.word2idx.get('<BOS>', 0)]  # Start with BOS
        for token in tokens:
            if len(token_ids) == self.max_length-1:
                break
            token_ids.append(self.word2idx.get(token, self.word2idx['<UNK>']))  # Replace unknown words
            
        token_ids.append(self.word2idx.get('<EOS>', 0))  # End with EOS
        if len(token_ids) < self.max_length:
            token_ids += [self.word2idx.get('<PAD>')] * (self.max_length-len(token_ids))
        return token_ids

    def encode_keywords(self, text):

        tokens = text.split()
        token_ids = []  
        for token in tokens:
            if len(token_ids) == self.max_length:
                break
            token_ids.append(self.word2idx.get(token, self.word2idx['<UNK>']))  # Replace unknown words
        return token_ids
    
    def decode(self, token_ids):
        """
        Convert a list of token indices back into text.
        Removes <BOS> and <EOS> tokens.
        """
        tokens = [self.idx2word.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join([token for token in tokens if token not in ['<BOS>', '<EOS>','<PAD>']])



# tokenizer = Tokenizer()
# filepaths = [
#     "data/DeepEyeNet_train.json",
#     "data/DeepEyeNet_val.json",
#     "data/DeepEyeNet_test.json"
# ]

# cleaned_filepaths = [
#     "data/cleaned_DeepEyeNet_train.json",
#     "data/cleaned_DeepEyeNet_val.json",
#     "data/cleaned_DeepEyeNet_test.json"
# ]

# print("🔎 Loading and cleaning text...")
# all_texts, cleaned_data = tokenizer.collect_texts(filepaths)

# print("📊 Building vocabulary...")
# tokenizer.build_vocab(all_texts)

# # Save vocabulary
# vocab_path = "data/vocab.json"
# tokenizer.save_vocab(vocab_path)
# print(f"✅ Vocabulary saved to {vocab_path}")

# # ===========================
# # Step 2: Replace Rare Words & Write Cleaned Data
# # ===========================

# print("\n📝 Replacing rare words and saving cleaned files...")

# for orig_path, out_path in zip(filepaths, cleaned_filepaths):
#     anns, cleaned_entries = cleaned_data[orig_path]
    
#     print(f"Processing {orig_path} -> {out_path}...")

#     idx = 0
#     for item in anns:
#         for meta in item.values():
#             # Replace rare words in keywords
#             cleaned_keywords = cleaned_entries[idx]['keywords']
#             if cleaned_keywords:
#                 cleaned_kw_list = [tokenizer.replace_rare(kw) for kw in cleaned_keywords.split(', ')]
#                 meta['keywords'] = ', '.join(cleaned_kw_list)
            
#             # Replace rare words in clinical-description
#             cleaned_description = cleaned_entries[idx]['clinical-description']
#             if cleaned_description:
#                 meta['clinical-description'] = tokenizer.replace_rare(cleaned_description)
#                 meta['original'] = cleaned_entries[idx]['original']
            
#             idx += 1
    
#     # Write to cleaned file
#     with open(out_path, 'w') as f:
#         json.dump(anns, f, indent=2)

#     print(f"✅ Cleaned data saved to {out_path}")

# print("\n🎉 All files processed successfully!")

