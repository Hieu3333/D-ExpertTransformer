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
args = parser_arg()
keywords = load_all_keywords()
tokenizer = Tokenizer()
tokenizer.load_vocab("vocab.json")

# Initialize dataset and dataloader
train_dataloader = DENDataLoader(args, tokenizer, keywords, split='train',shuffle=True)
val_dataloader = DENDataLoader(args,tokenizer,keywords,split='val',shuffle=False)
test_dataloader = DENDataLoader(args,tokenizer,keywords,split='test',shuffle=False)

