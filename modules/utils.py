import argparse
import json
# from torch.vision import transforms
def parser_arg():
    parser = argparse.ArgumentParser('Training script')
    parser.add_argument('--exp_name',type=str, required=True)
    parser.add_argument('--epochs',type=int, required=True)
    parser.add_argument('--batch_size',type=int, required=True)
    parser.add_argument('--max_length',type=int, required=True)
    parser.add_argument('--dataset',type=str, required=True)
    parser.add_argument('--max_gen',type=int, required=True)
    parser.add_argument('--hidden_size',type=int, required=True)
    parser.add_argument('--contrastive_proj_dim',type=int, required=True)
    parser.add_argument('--fc_size',type=int, required=True)
    parser.add_argument('--vocab_size',type=int, required=True)
    parser.add_argument('--ve_name',type=str, required=True)
    parser.add_argument('--use_diff',action='store_true')
    parser.add_argument('--randaug',action='store_true')
    parser.add_argument('--use_gca',action='store_true')
    parser.add_argument('--use_beam',action='store_true')
    parser.add_argument('--use_learnable_tokens',action='store_true')
    parser.add_argument('--constant_lr',action='store_true')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--use_contrastive',action='store_true')
    parser.add_argument('--channel_reduction',type=int, required=True)
    parser.add_argument('--num_layers',type=int, required=True)
    parser.add_argument('--warmup_epochs',type=int, required=True)
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
    parser.add_argument('--device',type=str, default='cuda')
    parser.add_argument('--accum_steps',type=int,required=True)
    parser.add_argument('--early_stopping',type=int,default=10)
    parser.add_argument('--project_root',type=str,required=True)
    parser.add_argument('--lambda_init',type=float, required=True)
    parser.add_argument('--from_pretrained',type=str,default=None)
    args, unparsed = parser.parse_known_args()
    return args
    

def get_inference_transform(args):
    if args.ve_name == "resnet" or args.ve_name == "densenet":
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    elif args.ve_name == "efficientnet":
        transform = transforms.Compose([
            transforms.Resize((356, 356)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
    else:
        raise ValueError(f"Unsupported backbone: {args.ve_name}")
    return transform
