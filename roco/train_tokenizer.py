from tokenizers import Tokenizer, models, trainers, pre_tokenizers, normalizers
from tokenizers.processors import TemplateProcessing
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence


# Initialize the tokenizer with BPE
tokenizer = Tokenizer(models.BPE())

# Normalization and pre-tokenization
tokenizer.normalizer = normalizers.Sequence([
    NFD(),
    Lowercase(),
    StripAccents()
])
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

# Define new special tokens
special_tokens = ["<PAD>", "<UNK>", "<BOS>", "<EOS>"]

# Trainer setup
trainer = trainers.BpeTrainer(
    vocab_size=20000,
    min_frequency=2,
    special_tokens=special_tokens
)

# Train from captions file
tokenizer.train(["captions.txt"], trainer)

# Set post-processing with BOS and EOS tokens
tokenizer.post_processor = TemplateProcessing(
    single="<BOS> $A <EOS>",
    pair="<BOS> $A <EOS> $B:1 <EOS>:1",  # optional if you plan on using pairs
    special_tokens=[
        ("<BOS>", tokenizer.token_to_id("<BOS>")),
        ("<EOS>", tokenizer.token_to_id("<EOS>"))
    ]
)


# Save the tokenizer
tokenizer.save("tokenizer.json")
print("✅ Tokenizer trained with <BOS>/<EOS> and saved!")
