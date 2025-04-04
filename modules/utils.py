import argparse
import json
def parser_arg():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--exp_name',type=str, required=True)
    parser.add_argument('--epochs',type=int, required=True)
    parser.add_argument('--batch_size',type=int, required=True)
    parser.add_argument('--max_length',type=int, required=True)
    parser.add_argument('--dataset',type=str, required=True)
    parser.add_argument('--max_gen',type=int, required=True)
    parser.add_argument('--hidden_size',type=int, required=True)
    parser.add_argument('--fc_size',type=int, required=True)
    parser.add_argument('--vocab_size',type=int, required=True)
    parser.add_argument('--ve_name',type=str, required=True)
    parser.add_argument('--channel_reduction',type=int, required=True)
    parser.add_argument('--num_layers',type=int, required=True)
    parser.add_argument('--step_size',type=int, required=True)
    parser.add_argument('--lr_ve',type=float, required=True)
    parser.add_argument('--lr_ed',type=float, required=True)
    parser.add_argument('--delta1',type=float, required=True)
    parser.add_argument('--delta2',type=float, required=True)
    parser.add_argument('--beam_width',type=int,default=1)
    parser.add_argument('--dropout',type=float, required=False)
    parser.add_argument('--topk',type=int, required=True)
    parser.add_argument('--temperature',type=float, required=True)
    parser.add_argument('--encoder_size',type=int, required=True)
    parser.add_argument('--num_heads',type=int, required=True)
    parser.add_argument('--diff_num_heads',type=int, required=True)
    parser.add_argument('--bias', action='store_true')
    parser.add_argument('--freeze_ve', action='store_true')
    parser.add_argument('--num_workers',type=int, default=8)
    parser.add_argument('--weight_decay',type=float, required=True)
    parser.add_argument('--log_interval',type=int, default=500)
    parser.add_argument('--save_path',type=str, required=True)
    parser.add_argument('--image_path',type=str, required=True)
    parser.add_argument('--ann_path',type=str, required=True)
    parser.add_argument('--use_diff',action='store_true')
    parser.add_argument('--randaug',action='store_true')
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--accum_steps',type=int,required=True)
    parser.add_argument('--early_stopping',type=int,default=10)
    parser.add_argument('--project_root',type=str,required=True)
    parser.add_argument('--lambda_init',type=float, required=True)
    parser.add_argument('--from_pretrained',type=str,default=None)
    args, unparsed = parser.parse_known_args()
    return args
    

def load_all_keywords():
    train_path = "data/cleaned_DeepEyeNet_train.json"
    val_path = "data/cleaned_DeepEyeNet_val.json"
    test_path = "data/cleaned_DeepEyeNet_test.json"
    all_keywords = set()
    json_paths = [train_path,val_path,test_path]
    for path in json_paths:
        print(f"Loading {path}...")
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        for item in data:
            for _, value in item.items():
                if value.get('keywords'):
                    keywords = value['keywords']
                    # Split by comma and strip whitespace
                    keyword_list = [kw.strip() for kw in keywords.split(',')]
                    all_keywords.update(keyword_list)
    
    # all_keywords = [kw for kw in all_keywords if "<UNK>" not in kw]
    print(f"Total {len(all_keywords)} keywords collected!")
    all_keywords = sorted(list(all_keywords))
    # print(all_keywords)
    return all_keywords