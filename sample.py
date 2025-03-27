from torchvision import transforms
from PIL import Image
import torch
from model.model import ExpertTransformer
import torch
from modules.tokenizer import Tokenizer
from model.model import ExpertTransformer
from modules.dataloader import DENDataLoader
from modules.metrics import compute_scores
from tqdm import tqdm
import os
from modules.utils import parser_arg, load_all_keywords
import torch.optim as optim

import logging
import random
import numpy as np


def preprocess_for_generation(image_path, keywords, tokenizer, transform):
    """
    Preprocesses an image and keywords to be used for caption generation.
    
    Args:
        image_path (str): Path to the image file.
        keywords (str): Keywords related to the image (comma-separated).
        tokenizer: Tokenizer used in the dataset.
        keywords_vocab_set: Set of known keywords.
        transform: Image transformation pipeline.

    Returns:
        torch.Tensor: Preprocessed image tensor (1, C, H, W).
        torch.Tensor: Encoded keyword tokens.
    """

    # --- Load and transform image ---
    image = Image.open(image_path).convert("RGB")  # Ensure RGB mode
    image_tensor = transform(image)  # Apply the same transform
    image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension

    # --- Process keywords ---
    keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
    
    if keywords_list:
        raw_keywords = " <SEP> ".join(keywords_list)  # Join keywords with <SEP>
    else:
        raw_keywords = "<SEP>"  # Handle empty keywords case

    keyword_tokens = tokenizer.encode(raw_keywords)  # Encode keywords
    keyword_tokens = keyword_tokens[:50]  # Truncate if too long
    keyword_tokens = torch.tensor(keyword_tokens, dtype=torch.long).unsqueeze(0)  # Add batch dim

    return image_tensor, keyword_tokens


args = parser_arg()

# Load all keywords
keywords = load_all_keywords()

# Load custom tokenizer
tokenizer = Tokenizer()
tokenizer.load_vocab("vocab.json")
# === Usage Example ===
# Define the same transform used in DeepEyeNet
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])

image_path = "data/eyenet0420/test_set/albinism-12.jpg"
keywords = "albinism"

# Assume `tokenizer` and `keywords_vocab_set` are already defined in your dataset
image_tensor, keyword_tokens = preprocess_for_generation(image_path, keywords, tokenizer, transform)
image_tensor, keyword_tokens = image_tensor.to(args.device), keyword_tokens.to(args.device)

print("Image Tensor Shape:", image_tensor.shape)  # Expected: (1, 3, 224, 224)
print("Keyword Tokens:", keyword_tokens)


model = ExpertTransformer(args,tokenizer,keywords)
checkpoint_path = os.path.join(args.project_root,"results/checkpoint_epoch_100.pth")
checkpoint = torch.load(checkpoint_path,map_location=args.device)
model.load_state_dict(checkpoint['model'])

model = model.to(args.device)

# Call the model's generate function
generated_caption = model.generate(image_tensor, keyword_tokens)
print("Generated Caption:", generated_caption)
