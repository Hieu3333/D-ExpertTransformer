import json
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import copy

class IUXray(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.max_length = args.max_length
        self.split = split  # 'train', 'test', 'val'
        self.tokenizer = tokenizer
        self.transform = transform

        # Set paths
        project_root = args.project_root
        ann_file = f'cleaned_iu_xray_{split}.json'
        self.ann_path = os.path.join(project_root, args.ann_path, ann_file)
        image_folder = f'{split}_set'
        self.image_path = os.path.join(project_root, 'data/images', image_folder)

        # Load annotations
        with open(self.ann_path, 'r') as f:
            self.annotations = json.load(f)

        # Flatten the list of dicts into (image_path, cleaned_report)
        self.data = []
        for entry in self.annotations:
            img_path = entry['image_path']
            cleaned_report = entry['cleaned_report']
            self.data.append((img_path, cleaned_report))

        # Define special token IDs
        self.pad_token_id = self.tokenizer.word2idx["<PAD>"]
        self.eos_token_id = self.tokenizer.word2idx["<EOS>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name, cleaned_report = self.data[idx]

        # Load Image
        full_img_path = os.path.join(self.image_path, os.path.basename(img_name))
        image_id = os.path.splitext(os.path.basename(img_name))[0]
        image = Image.open(full_img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)

        # --- Tokenize cleaned report --- (using tokenizer)
        report_tokens = self.tokenizer.encode(cleaned_report)

        # --- Create target tokens (shift left) --- 
        report_tokens = torch.tensor(report_tokens, dtype=torch.long)
        target_tokens = copy.deepcopy(report_tokens)
        target_tokens[:-1] = report_tokens[1:]  # Shift left
        target_tokens[-1] = self.pad_token_id  # Set last token as <PAD>

        # Return image, report tokens, and target tokens
        return image_id, image, report_tokens, target_tokens, cleaned_report
