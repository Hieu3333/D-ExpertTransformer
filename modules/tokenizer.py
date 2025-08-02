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
        all_cleaned_text = []
        data_store = {}  
        
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

        for text in all_texts:
            self.counter.update(text.split())

        for word, freq in self.counter.items():
            if freq > 1: 
                idx = len(self.word2idx)
                self.word2idx[word] = idx
                self.idx2word[idx] = word
            else:
                print(word)


    def replace_rare(self, text):
        words = text.split()
        return ' '.join(w if w in self.word2idx else '<UNK>' for w in words)

    def save_vocab(self, filepath):
        with open(filepath, 'w') as f:
            json.dump(self.word2idx, f, indent=2)

    def load_vocab(self, filepath):
        with open(filepath, 'r') as f:
            self.word2idx = json.load(f)
            self.idx2word = {idx: word for word, idx in self.word2idx.items()}
    
    def encode(self, text):

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

        tokens = [self.idx2word.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join([token for token in tokens if token not in ['<BOS>', '<EOS>','<PAD>']])



