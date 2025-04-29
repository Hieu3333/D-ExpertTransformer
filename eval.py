import torch
from modules.tokenizer import Tokenizer
from model.model import ExpertTransformer
from modules.dataloader import DENDataLoader
from modules.metrics import compute_scores
from tqdm import tqdm
import os
from modules.utils import parser_arg, get_inference_transform
import torch.optim as optim

import logging
import random
import numpy as np
import json
from torchvision import transforms
from PIL import Image


def set_seed(seed=42):
    random.seed(seed)  
    np.random.seed(seed)  
    torch.manual_seed(seed)  
    torch.cuda.manual_seed(seed)  
    torch.cuda.manual_seed_all(seed)  

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True  
    torch.backends.cudnn.benchmark = False  


    # # Extra safety: Ensure deterministic behavior for NumPy and PyTorch operations
    # torch.use_deterministic_algorithms(True)  # Enforces full determinism in PyTorch >=1.8
    # os.environ["PYTHONHASHSEED"] = str(seed)  # Ensures reproducibility for Python hash-based operations

# torch.set_float32_matmul_precision('high')
# Set the seed before training
set_seed(2003)
# Configure logger
logger = logging.getLogger("TrainingLogger")
logger.setLevel(logging.INFO)  # Change to DEBUG for more details

# Formatter
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# (Optional) File handler to save logs to file



# Parse arguments (ensure parser_arg() is defined appropriately)
args = parser_arg()


# Load all keywords


# Load custom tokenizer
tokenizer = Tokenizer(args)
tokenizer.load_vocab("data/vocab.json")


# Initialize model
model = ExpertTransformer(args, tokenizer)


optimizer =model.configure_optimizer(args)





if args.from_pretrained is not None:
    checkpoint_path = os.path.join(args.project_root,args.from_pretrained)
    checkpoint = torch.load(checkpoint_path,map_location=args.device)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    current_epoch = checkpoint['epoch']
    for param_id, param_state in optimizer.state.items():
        for key, value in param_state.items():
            if isinstance(value, torch.Tensor):
                param_state[key] = value.to(args.device)


# Define device
device = args.device
model.to(device)


total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])

path = ''
transform = get_inference_transform(args)
image = Image.open(path).convert('RGB')
input = transform(image)
input = input.unsqueeze(0)
input = input.to(device)

model.eval()
with torch.no_grad():
    output = model.generate_beam(input)

print(output)












