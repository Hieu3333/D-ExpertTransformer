import re
import json
from collections import Counter

class Tokenizer:
    def __init__(self, args=None):
        self.counter = Counter()
        self.special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        self.word2idx = {token: idx for idx, token in enumerate(self.special_tokens)}
        self.idx2word = {idx: token for token, idx in self.word2idx.items()}
        if args is not None:
            self.max_length = args.max_length
        else:
            self.max_length = 50

    def clean_text(self, report):
        """Clean the report text."""
        report_cleaner = lambda t: t.replace('..', '.').replace('..', '.').replace('..', '.').replace('1. ', '') \
            .replace('. 2. ', '. ').replace('. 3. ', '. ').replace('. 4. ', '. ').replace('. 5. ', '. ') \
            .replace(' 2. ', '. ').replace(' 3. ', '. ').replace(' 4. ', '. ').replace(' 5. ', '. ') \
            .strip().lower().split('. ')
        sent_cleaner = lambda t: re.sub('[.,?;*!%^&_+():-\[\]{}]', '', t.replace('"', '').replace('/', '').
                                        replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) + ' .'
        return report

    def collect_reports(self, data):
        """Collect all reports from the dataset for building vocab."""
        all_texts = []
        for split in ["train", "val", "test"]:
            for entry in data[split]:
                report = entry["report"]
                cleaned_report = self.clean_text(report)
                entry["cleaned_report"] = cleaned_report  # Add cleaned report to entry
                all_texts.append(cleaned_report)
        return data, all_texts  # Return modified data and all cleaned texts

    def build_vocab(self, all_texts):
        """Count words and build vocabulary."""
        for text in all_texts:
            self.counter.update(text.split())

        # Replace rare words (words appearing only once) with <UNK>
        for word, freq in self.counter.items():
            idx = len(self.word2idx)
            self.word2idx[word] = idx
            self.idx2word[idx] = word

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
            if len(token_ids) == self.max_length - 1:
                break
            token_ids.append(self.word2idx.get(token, self.word2idx['<UNK>']))  # Replace unknown words
            
        token_ids.append(self.word2idx.get('<EOS>', 0))  # End with EOS
        if len(token_ids) < self.max_length:
            token_ids += [self.word2idx.get('<PAD>')] * (self.max_length - len(token_ids))
        return token_ids

    def decode(self, token_ids):
        """
        Convert a list of token indices back into text.
        Removes <BOS> and <EOS> tokens.
        """
        tokens = [self.idx2word.get(idx, '<UNK>') for idx in token_ids]
        return ' '.join([token for token in tokens if token not in ['<BOS>', '<EOS>', '<PAD>']])



# path = "iu_xray/annotations.json"
# with open(path, 'r') as f:
#     data = json.load(f)

# tokenizer = Tokenizer()

# # Collect all reports and clean them
# cleaned_data, all_reports = tokenizer.collect_reports(data)

# # Build vocabulary from the reports
# tokenizer.build_vocab(all_reports)

# # Save the vocabulary to a file
# tokenizer.save_vocab('iu_xray/vocab.json')

# # Save cleaned data to a new JSON file
# cleaned_file_path = 'iu_xray/cleaned_iu_xray.json'
# with open(cleaned_file_path, 'w') as f:
#     json.dump(cleaned_data, f, indent=2)

# print(f"Cleaned data saved to {cleaned_file_path}")
# print("Vocabulary saved to iu_xray/vocab.json")

# text = "the heart size and pulmonary vascularity appear within normal limits"
# print(tokenizer.decode(tokenizer.encode(text)))
