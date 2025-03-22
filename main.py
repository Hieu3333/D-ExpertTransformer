import torch
from tokenizers import Tokenizer
from model.model import ExpertTransformer
from modules.dataloader import DENDataLoader
from modules.utils import parser_arg, load_all_keywords



    

args = parser_arg()
keywords = load_all_keywords()
tokenizer = Tokenizer.from_file('modules/custom_tokenizer.json')
dataloader = DENDataLoader(args,tokenizer,keywords,split='train',shuffle=False)
print(dataloader)

# Fetch one batch to inspect
for batch in dataloader:
    images, desc_tokens, one_hot, image_ids = batch  # Assuming you modified __getitem__ to return image_id
    print("Images shape:", images.shape)            # Typically (batch_size, 3, 224, 224)
    print("Desc Tokens shape:", desc_tokens.shape)  # (batch_size, max_length)
    print("One-hot labels shape:", one_hot.shape)   # (batch_size, num_keywords)
    print("Image IDs:", image_ids)                  # List of image names
    break  # Only show first batch

# import sys
# print(sys.path)