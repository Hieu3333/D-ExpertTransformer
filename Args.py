from dataclasses import dataclass
import torch

@dataclass
class Args:
    epochs: int = 50
    batch_size: int = 64
    max_length: int = 50
    dataset: str = "deepeyenet"
    max_gen: int = 100
    hidden_size: int = 512
    contrastive_proj_dim: int = 256
    fc_size: int = 2048
    vocab_size: int = 2690
    ve_name: str = "densenet"
    randaug: bool = False
    use_gca: bool = True
    use_learnable_tokens: bool = False
    constant_lr: bool = False
    eval: bool = False
    use_contrastive: bool = False
    channel_reduction: int = 4
    num_layers: int = 3
    warmup_epochs: int = 0
    step_size: int = 10
    lr_ve: float = 1e-4
    lr_ed: float = 2e-4
    delta1: float = 1.0
    delta2: float = 0.3
    beam_width: int = 3
    dropout: float = 0.2
    topk: int = 3
    temperature: float = 1.0
    encoder_size: int = 1024
    num_heads: int = 8
    diff_num_heads: int = 4
    bias: bool = False
    freeze_ve: bool = False
    num_workers: int = 16
    weight_decay: float = 1e-4
    log_interval: int = 500
    save_path: str = "results"
    image_path: str = "data/eyenet0420"
    ann_path: str = "data"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    accum_steps: int = 1
    early_stopping: int = 10
    project_root: str = "/mnt/c/D-ExpertTransformer"
    lambda_init: float = 0.8
    from_pretrained: str = "results/gca/deepeyenet/densenet_deepeyenet.pth"
