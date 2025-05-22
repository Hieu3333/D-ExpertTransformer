import json
import matplotlib.pyplot as plt
import numpy as np


def load_annotations(filepaths):
    all_data = []
    for path in filepaths:
        with open(path, 'r') as f:
            content = json.load(f)
            all_data.extend(content)
    return all_data

# File paths to your JSON files
filepaths = ['data/DeepEyeNet_train.json', 'data/DeepEyeNet_val.json', 'data/DeepEyeNet_test.json']  # Change to your actual file names
data = load_annotations(filepaths)
# Extract lengths
clinical_lengths = []
keyword_lengths = []

for item in data:
    for _, annotation in item.items():
        clinical_text = annotation["clinical-description"]
        keyword_text = annotation["keywords"]

        clinical_lengths.append(len(clinical_text.split()))
        keyword_lengths.append(len(keyword_text.split()))  # count keywords (separated by commas)

# Count and normalize
max_len = 80
bins = range(max_len + 1)

clinical_counts, _ = np.histogram(clinical_lengths, bins=bins)
keyword_counts, _ = np.histogram(keyword_lengths, bins=bins)

clinical_percent = 100 * clinical_counts / len(clinical_lengths)
keyword_percent = 100 * keyword_counts / len(keyword_lengths)

# Plot
plt.figure(figsize=(8, 5))
x = bins[:-1]

plt.plot(x, clinical_percent, color='blue', label='Clinical Description Length Distribution')
plt.plot(x, keyword_percent, color='red', label='Keyword Length Distribution')

plt.xlabel('Length (Words)')
plt.ylabel('Percentage %')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
