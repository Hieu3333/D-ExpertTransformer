from datasets import load_dataset
import torch

# Load the dataset once
ds = load_dataset("mdwiratathya/ROCO-radiology")

# Save the dataset to a file
torch.save(ds, 'roco_dataset.pt')
print("Dataset saved to roco_dataset.pt")
