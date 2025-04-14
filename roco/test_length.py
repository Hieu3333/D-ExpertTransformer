import matplotlib.pyplot as plt
from tokenizers import Tokenizer as HFTokenizer

# Load tokenizer
tokenizer = HFTokenizer.from_file("roco/tokenizer.json")

# Read captions
with open("roco/captions.txt", "r", encoding="utf-8") as f:
    captions = [line.strip() for line in f if line.strip()]

# Compute token lengths
caption_lengths = [len(tokenizer.encode(caption).ids) for caption in captions]

# Plot distribution
plt.figure(figsize=(10, 6))
plt.hist(caption_lengths, bins=30, color='skyblue', edgecolor='black')
plt.title("Caption Length Distribution")
plt.xlabel("Number of Tokens")
plt.ylabel("Number of Captions")
plt.grid(True)
plt.tight_layout()
plt.show()
