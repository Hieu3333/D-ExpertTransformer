from iu_xray.tokenizer import Tokenizer
from modules.utils import parser_arg
from iu_xray.dataloader import IUXrayDataLoader
from tqdm import tqdm

# Load your trained tokenizer
args = parser_arg()
tokenizer = Tokenizer(args)
tokenizer.load_vocab('iu_xray/vocab.json')

train = IUXrayDataLoader(args,tokenizer,split="train",shuffle=False)

# Test cases

for batch_idx, batch in enumerate(tqdm(train, desc=f"Epoch")):
    image_ids, images, desc_tokens, target_tokens, gt_clinical_desc = batch
    print("id", image_ids)
    print('images:', images.shape)
    print("desc:", desc_tokens)
    print("target:",target_tokens)