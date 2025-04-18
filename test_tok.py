from modules.utils import get_mask_prob
import torch


for i in range(1,51):
    y = get_mask_prob(50,i)
    print(f"{i} - {y}")