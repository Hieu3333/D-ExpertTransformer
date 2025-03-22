import torch
from tokenizers import Tokenizer
from model.model import ExpertTransformer
from modules.dataloader import DENDataLoader
from modules.metrics import compute_scores
from modules.utils import parser_arg, load_all_keywords



    

args = parser_arg()
keywords = load_all_keywords()
tokenizer = Tokenizer.from_file('modules/custom_tokenizer.json')
dataloader = DENDataLoader(args,tokenizer,keywords,split='train',shuffle=False)
device = 'cpu'
model = ExpertTransformer(args,tokenizer,keywords)
model = model.to(device)



# for batch in dataloader:
#     image_id, images, desc_tokens, target_tokens, one_hot = batch
#     # print("desc_tokens:",desc_tokens)
#     # print("target_tokens:",target_tokens)
#     images, desc_tokens, target_tokens, one_hot =  images.to(device), desc_tokens.to(device), \
#           target_tokens.to(device), one_hot.to(device)
#     output,loss = model(images,desc_tokens,target_tokens, one_hot)
#     print('Output:',output.shape)
#     break




