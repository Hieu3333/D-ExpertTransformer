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
        ann_file = 'cleaned_iu_xray.json'
        

        self.ann_path = os.path.join(project_root, args.ann_path, ann_file)
        self.image_path = os.path.join(project_root, 'data/iu_xray/images')

        # Load annotations
        with open(self.ann_path, 'r') as f:
            self.annotations = json.load(f)

        # Flatten the list of dicts into (image_path, cleaned_report)
        self.data = []
        for entry in self.annotations[split]:
            img_path = entry['image_path']
            cleaned_report = entry['cleaned_report']
            image_id = entry['id']
            self.data.append((image_id,img_path, cleaned_report))

        # Define special token IDs
        self.pad_token_id = self.tokenizer.word2idx["<PAD>"]
        self.eos_token_id = self.tokenizer.word2idx["<EOS>"]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_id, img_names, cleaned_report = self.data[idx]

        # --- Load Both Images ---
        images = []
        for img_name in img_names:  
            full_img_path = os.path.join(self.image_path, img_name)
            image = Image.open(full_img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            images.append(image)

        # Stack to tensor: ( C, 2*H, W)
        images = torch.stack(images, dim=1)
        images = images.view(images.size(0),-1,images.size(-1))

        # --- Tokenize cleaned report ---
        report_tokens = self.tokenizer.encode(cleaned_report)
        report_tokens = torch.tensor(report_tokens, dtype=torch.long)

        # --- Create target tokens (shifted) ---
        target_tokens = copy.deepcopy(report_tokens)
        target_tokens[:-1] = report_tokens[1:]
        target_tokens[-1] = self.pad_token_id

        return image_id, images, report_tokens, target_tokens, cleaned_report

