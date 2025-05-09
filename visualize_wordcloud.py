import json
import os
import re
from collections import Counter
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

# Add any extra stopwords if needed
EXTRA_STOPWORDS = set([
    'with', 'of', 'and', 'or', 'in', 'the', 'a', 'to', 'for', 'on', 'by',
    'is', 'as', 'from', 'was', 'are', 'at', 'an', 'be', 'that', 'this',
    'we', 'which', 'were', 'also', 'has', 'had', 'have', 'but'
])

def clean_and_tokenize(text, extra_stopwords=set()):
    # Lowercase, remove punctuation, and split
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()

    # Remove stopwords and numbers
    stopwords = STOPWORDS.union(extra_stopwords)
    words = [word for word in words if word not in stopwords and word.isalpha()]

    return words

def load_and_count_words(filenames):
    word_counter = Counter()

    for file in filenames:
        with open(file, 'r') as f:
            data = json.load(f)

        for entry in data:
            for v in entry.values():
                clinical_text = v.get("clinical-description", "")
                keyword_text = v.get("keywords", "")

                clinical_words = clean_and_tokenize(clinical_text, EXTRA_STOPWORDS)
                keyword_words = clean_and_tokenize(keyword_text, EXTRA_STOPWORDS)

                word_counter.update(clinical_words + keyword_words)

    return word_counter

# List your json files here
filepaths = ['data/DeepEyeNet_train.json', 'data/DeepEyeNet_val.json', 'data/DeepEyeNet_test.json']  

# Count words
word_counts = load_and_count_words(filepaths)
top_100 = dict(word_counts.most_common(100))
# Generate word cloud
wordcloud = WordCloud(width=800, height=400, background_color='white').generate_from_frequencies(top_100)

# Show word cloud
plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.tight_layout()
plt.show()
