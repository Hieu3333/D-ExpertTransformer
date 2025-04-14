import torch
from datasets import load_dataset
from torch.utils.data import Dataset
import copy
import re
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence




class ROCO(Dataset):
    def __init__(self, args, data,tokenizer, transform=None, split='train'):
        self.args = args
        self.transform = transform
        self.split = split
        self.data = data[split]  # Hugging Face DatasetDict split (e.g., 'train', 'test', etc.)
        self.tokenizer = tokenizer
        self.normalizer = Sequence([
                            NFD(),
                            Lowercase(),
                            StripAccents()
                        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        image = sample['image']         # PIL Image
        caption = sample['caption']     # String
        image_id = sample.get('image_id', None)  # Optional

        normalized_caption = self.normalizer.normalize_str(caption)
        normalized_caption = normalized_caption.strip().replace('\n',' ')
        # Safely remove broken Unicode escape sequences (skip decode to avoid crash)
        normalized_caption = re.sub(r'\\u[0-9a-fA-F]{0,4}', '', normalized_caption)
        normalized_caption = re.sub(r'[^a-zA-Z0-9\s]', '', normalized_caption)  # keep only alphanum, space, and ()
        normalized_caption = re.sub(r'\s+', ' ', normalized_caption).strip() 

        tokens = self.tokenizer.encode(normalized_caption)
        tokens = torch.tensor(tokens,dtype=torch.long)
        target_tokens = copy.deepcopy(tokens)
        target_tokens[:-1] = tokens[1:]
        target_tokens[-1] = self.tokenizer.pad_id

   

        # Apply transforms if provided
        if self.transform:
            image = image.convert("RGB")
            image = self.transform(image)
        
     
        return image_id,image, tokens,target_tokens,normalized_caption
