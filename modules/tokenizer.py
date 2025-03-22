import re
from collections import Counter
import json
import re
from collections import Counter

class Tokenizer:
    def __init__(self, min_freq=1):
        self.word2idx = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}
        self.idx2word = {0: '<PAD>', 1: '<UNK>', 2: '<BOS>', 3: '<EOS>'}
        self.min_freq = min_freq
        self.counter = Counter()
        self.important_words = ['re', 'le', 'od', 'os']

    def clean_report(self, report):
        text = re.sub(r'[\/\-\.]', ' ', report)  # Replace / - . with space
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)  # Keep letters, digits, space
        text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
        return text.lower().strip()

    def build_vocab(self, reports):
        for text in reports:
            cleaned = self.clean_report(text)
            words = cleaned.split()
            self.counter.update(words)
        idx = len(self.word2idx)
        for word, freq in self.counter.items():
            if freq >= self.min_freq or word in self.important_words:
                if word not in self.word2idx:
                    self.word2idx[word] = idx
                    self.idx2word[idx] = word
                    idx += 1
        # Force add important words
        for word in self.important_words:
            if word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1
        print("Vocab size:", len(self.word2idx))

    def encode(self, report):
        cleaned = self.clean_report(report)
        tokens = []
        for word in cleaned.split():
            if self.counter[word] >= self.min_freq or word in self.important_words:
                tokens.append(self.word2idx.get(word, self.word2idx['<UNK>']))
            else:
                tokens.append(self.word2idx['<UNK>'])
        return tokens

    def decode(self, indices):
        return ' '.join([self.idx2word.get(idx, '<UNK>') for idx in indices])


path = r"C:\Users\hieu3\Downloads\DeepEyeNet\DeepEyeNet_train.json"
anns = json.load(open(path))
all_text = []
for item in anns:
    for meta in item.values():
        keywords = meta.get('keywords','')
        description = meta.get('clinical_description','')
    if keywords:
        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        all_text.extend(keywords_list)
    if description:
        all_text.extend(description.strip())

path = r"C:\Users\hieu3\Downloads\DeepEyeNet\DeepEyeNet_valid.json"
anns = json.load(open(path))
for item in anns:
    for meta in item.values():
        keywords = meta.get('keywords','')
        description = meta.get('clinical_description','')
    if keywords:
        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        all_text.extend(keywords_list)
    if description:
        all_text.extend(description.strip())

path = r"C:\Users\hieu3\Downloads\DeepEyeNet\DeepEyeNet_test.json"
anns = json.load(open(path))
for item in anns:
    for meta in item.values():
        keywords = meta.get('keywords','')
        description = meta.get('clinical_description','')
    if keywords:
        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        all_text.extend(keywords_list)
    if description:
        all_text.extend(description.strip())

tokenizer = Tokenizer()
tokenizer.build_vocab(all_text)
text = "69-year-old white male. rpe tear and srnv-md. re 20/200 le 20/20."
print(tokenizer.decode(tokenizer.encode(text)))

