import json
import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import copy
from PIL import Image



class DeepEyeNet(Dataset):
    def __init__(self, args, tokenizer, keywords_list, split, transform=None):
        self.max_length = args.max_length
        self.split = split  # 'train', 'test', 'val'
        self.tokenizer = tokenizer
        self.transform = transform

        # --- Vocabulary ---
        self.keywords_vocab = keywords_list  # set → sorted list
        self.keyword_to_idx = {kw: idx for idx, kw in enumerate(self.keywords_vocab)}
        self.num_keywords = len(self.keywords_vocab)

        # Set paths
        project_root = args.project_root
        ann_file = f'cleaned_DeepEyeNet_{split}.json'
        self.ann_path = os.path.join(project_root, args.ann_path, ann_file)
        image_folder = f'{split}_set'
        self.image_path = os.path.join(project_root, 'data/eyenet0420', image_folder)

        # Load annotations
        with open(self.ann_path, 'r') as f:
            self.annotations = json.load(f)

        # Flatten the list of dicts into (image_path, keywords, clinical_description)
        self.data = []
        for entry in self.annotations:
            for img_path, content in entry.items():
                keywords = content.get('keywords', '')
                clinical_desc = content.get('clinical-description', '')
                self.data.append((img_path, keywords, clinical_desc))

        # Define special token IDs
        self.pad_token_id = self.tokenizer.word2idx["<PAD>"]
        self.eos_token_id = self.tokenizer.word2idx["<EOS>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, keywords, clinical_desc = self.data[idx]

        # --- One-hot encoding for keywords ---
        one_hot = torch.zeros(self.num_keywords, dtype=torch.float32)
        keywords_list = [kw.strip() for kw in keywords.split(',') if kw.strip()]
        for kw in keywords_list:
            if kw in self.keyword_to_idx:
                one_hot[self.keyword_to_idx[kw]] = 1.0

        # Load Image
        full_img_path = os.path.join(self.image_path, os.path.basename(img_name))
        image_id = os.path.splitext(os.path.basename(img_name))[0]
        image = Image.open(full_img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # --- Tokenize clinical description ---
        desc_tokens = self.tokenizer.encode(clinical_desc)
        

        # --- Create target tokens (shift left) ---
        desc_tokens = torch.tensor(desc_tokens, dtype=torch.long)
        target_tokens = copy.deepcopy(desc_tokens)
        target_tokens[:-1] = desc_tokens[1:]  # Shift left
        target_tokens[-1] = self.pad_token_id  # Set last token as <PAD>

        

        # --- Encode keywords with <SEP> separator ---
        if keywords_list:
            raw_keywords = f" <SEP> ".join(keywords_list)  # Join keywords with <SEP>
        else:
            raw_keywords = "<SEP>"  # Handle empty keywords case
        
        keyword_tokens = self.tokenizer.encode_keywords(raw_keywords)
        keyword_tokens = torch.tensor(keyword_tokens, dtype=torch.long)

        return image_id, image, desc_tokens, target_tokens, keyword_tokens, clinical_desc




