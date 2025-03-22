import argparse
import json
def parser_arg():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--batch_size',type=int, required=True)
    parser.add_argument('--max_length',type=int, required=True)
    parser.add_argument('--hidden_size',type=int, required=True)
    parser.add_argument('--vocab_size',type=int, required=True)
    parser.add_argument('--threshold',type=float, required=True)
    parser.add_argument('--num_layers',type=int, required=True)
    parser.add_argument('--delta1',type=float, required=True)
    parser.add_argument('--delta2',type=float, required=True)
    parser.add_argument('--dropout',type=float, required=False)
    parser.add_argument('--encoder_size',type=int, required=True)
    parser.add_argument('--decoder_size',type=int, required=True)
    parser.add_argument('--num_head',type=int, required=True)
    parser.add_argument('--keyword_vocab_size',type=int, required=True)
    parser.add_argument('--num_workers',type=int, default=8)
    parser.add_argument('--image_path',type=str, required=True)
    parser.add_argument('--ann_path',type=str, required=True)
    parser.add_argument('--randaug',type=bool, required=False)
    args, unparsed = parser.parse_known_args()
    return args
    

def load_all_keywords():
    train_path = "data/DeepEyeNet_train.json"
    val_path = "data/DeepEyeNet_valid.json"
    test_path = "data/DeepEyeNet_test.json"
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
    print(f"Total {len(all_keywords)} keywords collected!")
    print(all_keywords)
    return all_keywords