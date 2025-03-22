import torch
from tokenizers import Tokenizer
from model.model import ExpertTransformer
from modules.dataloader import DENDataLoader
from modules.metrics import compute_scores
from tqdm import tqdm
import os
from modules.utils import parser_arg, load_all_keywords
import torch.optim as optim


# Parse arguments (ensure parser_arg() is defined appropriately)
args = parser_arg()

# Load all keywords
keywords = load_all_keywords()

# Load custom tokenizer
tokenizer = Tokenizer.from_file('modules/custom_tokenizer.json')

# Initialize dataset and dataloader
train_dataloader = DENDataLoader(args, tokenizer, keywords, split='train',shuffle=True)
val_dataloader = DENDataLoader(args,tokenizer,keywords,split='val',shuffle=False)
test_dataloader = DENDataLoader(args,tokenizer,keywords,split='test',shuffle=False)

# Initialize model
model = ExpertTransformer(args, tokenizer, keywords)

# Define device
device = args.device
model.to(device)

optimizer = optim.Adam(model.parameters(), lr=args.lr)

# Training parameters
num_epochs = args.epochs
log_interval = args.log_interval
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print('Total params:',total_params)

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        image_ids, images, desc_tokens, target_tokens, one_hot = batch
        images, desc_tokens, target_tokens, one_hot = images.to(device), desc_tokens.to(device), target_tokens.to(device), one_hot.to(device)
        optimizer.zero_grad()
        outputs, loss = model(images, desc_tokens, target_tokens, one_hot)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
        if (batch_idx + 1) % log_interval == 0:
            avg_loss = running_loss / log_interval
            print(f"Batch {batch_idx + 1}/{len(train_dataloader)}, Loss: {avg_loss:.4f}")
            running_loss = 0.0

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(save_path, f"model_epoch_{epoch+1}.pth"))

    #Evaluation
    model.eval()
    gts = {}
    res = {}
    with torch.no_grad():
        
        for batch in tqdm(val_dataloader, desc="Evaluating"):
            image_ids, images, desc_tokens, target_tokens, one_hot = batch
            images = images.to(device)
            target_tokens = target_tokens.to(device)
            # print("Image:",images.shape)
            # print("target_tokens:",target_tokens.shape)
            
            # Generate captions for the whole batch
            generated_captions = model.generate(images)  # List of strings, length B
            
            # Decode ground truth captions
            for i, image_id in enumerate(image_ids):
                reference_caption = tokenizer.decode(target_tokens[i].cpu().numpy(), skip_special_tokens=True)
                gts[image_id] = [reference_caption]
                res[image_id] = [generated_captions[i]]  # Corresponding generated caption
        
        # Compute evaluation metrics
        eval_scores = compute_scores(gts, res)
        print(f"Epoch {epoch + 1} Evaluation Scores: {eval_scores}")






