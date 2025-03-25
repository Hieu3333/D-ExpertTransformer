import torch
from tokenizers import Tokenizer
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


# Set the seed before training
set_seed(18)
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
file_handler = logging.FileHandler('training.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


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

ve_params = list(map(id, model.visual_extractor.parameters()))
ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
optimizer =torch.optim.AdamW(
            [{'params': model.visual_extractor.parameters(), 'lr': args.lr_ve},
             {'params': ed_params, 'lr': args.lr_ed}],
            weight_decay=args.weight_decay
        )
if args.from_pretrained is not None:
    checkpoint = os.path.join(args.project_root,args.from_pretrained)
    model.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optim'])
    current_epoch = checkpoint['epoch']
else:
    current_epoch = 1

# Define device
device = args.device
model.to(device)
model = torch.compile(model)

# Training parameters
num_epochs = args.epochs
log_interval = args.log_interval
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print('Total params:',total_params)

num_epoch_not_improved = 0
best_epoch = 0
best_avg_bleu = 0

logger.info(args)



for epoch in range(current_epoch-1,num_epochs):
    if num_epoch_not_improved == args.early_stopping:
        break

    logger.info(f"Epoch {epoch+1}:")

    model.train()
    running_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        image_ids, images, desc_tokens, target_tokens, one_hot = batch
        images, desc_tokens, target_tokens, one_hot = images.to(device), desc_tokens.to(device), target_tokens.to(device), one_hot.to(device)

        outputs, loss, loss_ce = model(images, desc_tokens, target_tokens, one_hot)
        loss = loss / args.accum_steps  # Normalize for gradient accumulation

        loss.backward()

        # Gradient accumulation step
        if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1 == len(train_dataloader)):
            optimizer.step()
            optimizer.zero_grad()  # Zero gradients after step

        running_loss += loss.item()
        
        # Logging
        if (batch_idx + 1) % log_interval == 0 or batch_idx == 0 or (batch_idx+1== len(train_dataloader)) :
            avg_loss = running_loss / log_interval
            logger.info(f"Batch {batch_idx + 1}/{len(train_dataloader)} Loss: {avg_loss:.4f}")
            running_loss = 0.0  # Reset running loss
    

    


    #Evaluation
    model.eval()
    gts = {}
    res = {}
    with torch.no_grad():  
        for batch_idx,batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            image_ids, images, desc_tokens, target_tokens, one_hot = batch
            images = images.to(device)
            target_tokens = target_tokens.to(device)
            # print("Image:",images.shape)
            # print("target_tokens:",target_tokens.shape)
            
            # Generate captions for the whole batch
            # generated_captions = model.generate(images,beam_width=args.beam_width)  # List of strings, length B
            with torch.cuda.amp.autocast():
                generated_captions = model.generate(images)
            # Decode ground truth captions
            for i, image_id in enumerate(image_ids):
                groundtruth_caption = tokenizer.decode(target_tokens[i].cpu().numpy(), skip_special_tokens=True)
                gts[image_id] = [groundtruth_caption]
                res[image_id] = [generated_captions[i]]  # Corresponding generated caption
        
        # Compute evaluation metrics
        eval_scores = compute_scores(gts, res)
        logger.info(f"Epoch {epoch + 1} - Evaluation scores:")
        logger.info(f"BLEU_1: {eval_scores['BLEU_1']}")
        logger.info(f"BLEU_2: {eval_scores['BLEU_2']}")
        logger.info(f"BLEU_3: {eval_scores['BLEU_3']}")
        logger.info(f"BLEU_4: {eval_scores['BLEU_4']}")
        logger.info(f"METEOR: {eval_scores['METEOR']}")
        logger.info(f"CIDER: {eval_scores['Cider']}")
        logger.info(f"ROUGE_L: {eval_scores['ROUGE_L']}")
        

    model.eval()
    gts= {}
    res = {}
    with torch.no_grad():
        for batch_idx,batch in enumerate(tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            image_ids, images, desc_tokens, target_tokens, one_hot = batch
            images = images.to(args.device)
            target_tokens = target_tokens.to(device)
            # generated_captions = model.generate(images,beam_width=args.beam_width)
            with torch.cuda.amp.autocast():
                generated_captions = model.generate(images) 

        for i,image_id in enumerate(image_ids):
            groundtruth_caption = tokenizer.decode(target_tokens[i].cpu().numpy(),skip_special_tokens=True)
            gts[image_id] = [groundtruth_caption]
            res[image_id] = [generated_captions[i]]
        
        test_scores = compute_scores(gts,res)
        logger.info(f"Epoch {epoch + 1} - Test scores:")
        logger.info(f"BLEU_1: {test_scores['BLEU_1']}")
        logger.info(f"BLEU_2: {test_scores['BLEU_2']}")
        logger.info(f"BLEU_3: {test_scores['BLEU_3']}")
        logger.info(f"BLEU_4: {test_scores['BLEU_4']}")
        logger.info(f"METEOR: {test_scores['METEOR']}")
        logger.info(f"CIDER: {test_scores['Cider']}")
        logger.info(f"ROUGE_L: {test_scores['ROUGE_L']}")

    avg_bleu = (eval_scores['BLEU_1']+eval_scores['BLEU_2'])/2
    improved = avg_bleu > best_avg_bleu
    if not improved:
        num_epoch_not_improved += 1
        best_avg_bleu = avg_bleu
        best_epoch = epoch+1
    else:
        num_epoch_not_improved = 0
        torch.save({
            'epoch': epoch + 1,  # Save current epoch
            'model': model.state_dict(),  # Save model weights
            'optim': optimizer.state_dict(),  # Save optimizer state
        }, os.path.join(save_path, f"checkpoint_epoch_{epoch+1}.pth"))

        

    
    logger.info(f"Best epoch: {epoch+1}")
    logger.info("---------------------------------------------------------------------------------------------------------------------")







