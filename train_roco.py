import torch
from roco.tokenizer import Tokenizer
from model.model import ExpertTransformer
from roco.dataloader import ROCODataLoader
from modules.metrics import compute_scores
from tqdm import tqdm
import os
from modules.utils import parser_arg
import torch.optim as optim

import logging
import random
import numpy as np
import json
from torch.cuda.amp import autocast, GradScaler



def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

    # # Extra safety: Ensure deterministic behavior for NumPy and PyTorch operations
    # torch.use_deterministic_algorithms(True)  # Enforces full determinism in PyTorch >=1.8
    # os.environ["PYTHONHASHSEED"] = str(seed)  # Ensures reproducibility for Python hash-based operations

# torch.set_float32_matmul_precision('high')
# Set the seed before training
set_seed(18092003)
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
data = torch.load('roco_dataset.pt')
print("Dataset loaded from roco_dataset.pt")
# Load custom tokenizer
tokenizer = Tokenizer(args)

# Initialize dataset and dataloader
train_dataloader = ROCODataLoader(data=data,tokenizer=tokenizer, args=args, split='train',shuffle=True)
val_dataloader = ROCODataLoader(data=data,tokenizer=tokenizer, args=args, split='validation',shuffle=False)
test_dataloader = ROCODataLoader(data=data,tokenizer=tokenizer, args=args, split='test',shuffle=False)

# Initialize model
model = ExpertTransformer(args, tokenizer)
# torch.set_float32_matmul_precision('high')

optimizer =model.configure_optimizer(args)


scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs-args.warmup_epochs, eta_min=1e-6)


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
else:
    current_epoch = 1

# Define device
device = args.device
model.to(device)
model = torch.compile(model)

# Training parameters
num_epochs = args.epochs
log_interval = args.log_interval
save_path = os.path.join(args.save_path,args.exp_name)
log_path = os.path.join('roco/logs',args.exp_name)
os.makedirs(save_path, exist_ok=True)
os.makedirs(log_path, exist_ok=True)
total_params = sum([p.numel() for p in model.parameters() if p.requires_grad])
print(f'Total params: {total_params/1e6:.2f}M')

scaler = GradScaler()

logger.info(args)

best_test_score = 0.0
for epoch in range(current_epoch-1,num_epochs):
    logger.info(f"Epoch {epoch+1}:")

    model.train()
    running_loss = 0.0
    if not args.eval:
        for batch_idx, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}; lr={scheduler.get_last_lr()}")):
            image_ids, images, desc_tokens, target_tokens,captions = batch
            images, desc_tokens, target_tokens = images.to(device), desc_tokens.to(device), target_tokens.to(device)
            # print("desc_tokens:",desc_tokens)
            # print("target_tokens:",target_tokens)
            # print('gt:',gt_clinical_desc)
            with torch.cuda.amp.autocast():
                outputs, loss = model(images=images,tokens=desc_tokens, targets=target_tokens)
                loss = loss / args.accum_steps  # Normalize for gradient accumulation
            scaler.scale(loss).backward()
            
            # Gradient accumulation step
            if (batch_idx + 1) % args.accum_steps == 0 or (batch_idx + 1 == len(train_dataloader)):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                
                optimizer.zero_grad()  # Zero gradients after step
                scaler.step(optimizer)
                scaler.update()

            running_loss += loss.item()
            
            # Logging
            if (batch_idx+1== len(train_dataloader)) :
                avg_loss = running_loss / len(train_dataloader)
                logger.info(f"Batch {batch_idx + 1}/{len(train_dataloader)} Loss: {avg_loss:.4f} Norm: {norm:.2f}")
                running_loss = 0.0  # Reset running loss


    if not args.constant_lr and (epoch+1)>args.warmup_epochs:
        scheduler.step()  

    torch.save({
        'epoch': epoch + 1,  # Save current epoch
        'model': model.state_dict(),  # Save model weights
        'optim': optimizer.state_dict(),  # Save optimizer state
    }, os.path.join(save_path, f"roco.pth"))
    print(f'Model checkpoint saved to {save_path}')
        
    
    if (epoch+1) <args.epochs:
        continue
   
    
  
    val_results = []
    test_results = []

    #Evaluation
    model.eval()
    gts_val = {}
    res_val = {}
    val_loss = 0.0
    with torch.no_grad():  
        for batch_idx,batch in enumerate(tqdm(val_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            image_ids, images, desc_tokens, target_tokens,captions = batch
            
            images = images.to(device)
            target_tokens = target_tokens.to(device)
            desc_tokens = desc_tokens.to(device)
            # print("Image:",images.shape)
            # print("target_tokens:",target_tokens.shape)
            
            # Generate captions for the whole batch
            # generated_captions = model.generate(images,beam_width=args.beam_width)  # List of strings, length B
            with torch.cuda.amp.autocast():
                generated_captions = model.generate_beam(images)
            # Decode ground truth captions
            for i, image_id in enumerate(image_ids):
                groundtruth_caption = captions[i]
                gts_val[image_id] = [groundtruth_caption]
                res_val[image_id] = [generated_captions[i]]  # Corresponding generated caption
                val_results.append({"image_id": image_id, "ground_truth": groundtruth_caption, "generated_caption": generated_captions[i]})        


        val_path = os.path.join(log_path,f"val_result_epoch_{epoch+1}.json")
        with open(val_path, "w") as f:
            json.dump(val_results, f, indent=4)
        # Compute evaluation metrics
        eval_scores = compute_scores(gts_val, res_val)
  
        logger.info(f"Epoch {epoch + 1} - Evaluation scores:")
        logger.info(f"BLEU_1: {eval_scores['BLEU_1']}")
        logger.info(f"BLEU_2: {eval_scores['BLEU_2']}")
        logger.info(f"BLEU_3: {eval_scores['BLEU_3']}")
        logger.info(f"BLEU_4: {eval_scores['BLEU_4']}")
        logger.info(f"METEOR: {eval_scores['METEOR']}")
        logger.info(f"ROUGE_L: {eval_scores['ROUGE_L']}")
        print("GTS Val Example:", list(gts_val.items())[-5:-1])
        print("Res Val Example:", list(res_val.items())[-5:-1])
        logger.info(f"{eval_scores}")

    

    model.eval()
    gts_test= {}
    res_test = {}
    with torch.no_grad():
        for batch_idx,batch in enumerate(tqdm(test_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
            image_ids, images, desc_tokens, target_tokens,captions = batch
            images = images.to(args.device)
            target_tokens = target_tokens.to(device)
            # generated_captions = model.generate(images,beam_width=args.beam_width)
            with torch.cuda.amp.autocast():
                generated_captions = model.generate_beam(images) 

            for i,image_id in enumerate(image_ids):
                groundtruth_caption = captions[i]
                gts_test[image_id] = [groundtruth_caption]
                res_test[image_id] = [generated_captions[i]]
                test_results.append({"image_id": image_id, "ground_truth": groundtruth_caption, "generated_caption": generated_captions[i]})
        
        test_path = os.path.join(log_path,f"test_result_epoch_{epoch+1}.json")
        with open(test_path, "w") as f:
            json.dump(test_results, f, indent=4)

        
        
        test_scores = compute_scores(gts_test,res_test)
        logger.info(f"Epoch {epoch + 1} - Test scores:")
        logger.info(f"BLEU_1: {test_scores['BLEU_1']}")
        logger.info(f"BLEU_2: {test_scores['BLEU_2']}")
        logger.info(f"BLEU_3: {test_scores['BLEU_3']}")
        logger.info(f"BLEU_4: {test_scores['BLEU_4']}")
        logger.info(f"METEOR: {test_scores['METEOR']}")
        logger.info(f"ROUGE_L: {test_scores['ROUGE_L']}")

        

        print("GTS Test Example:", list(gts_test.items())[-5:-1])
        print("Res Test Example:", list(res_test.items())[-5:-1])

            
  
    logger.info("--------------------------------------------------------------------------------")





