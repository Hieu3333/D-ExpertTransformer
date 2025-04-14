import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from copy import deepcopy

class ROCO(Dataset):
    def __init__(self, args, data,tokenizer, transform=None, split='train'):
        self.args = args
        self.transform = transform
        self.split = split
        self.data = data[split]  # Hugging Face DatasetDict split (e.g., 'train', 'test', etc.)
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']         # PIL Image
        caption = sample['caption']     # String
        image_id = sample.get('image_id', None)  # Optional

        tokens = self.tokenizer.encode(caption)
        tokens = torch.tensor(tokens,dtype=torch.long)
        target_tokens = tokens.clone()

        target_tokens = tokens[1:]
        target_tokens[-1] = self.tokenizer.pad_id


        # Apply transforms if provided
        if self.transform:
            image = image.convert("RGB")
            image = self.transform(image)
     
        return image_id,image, tokens,target_tokens,caption
