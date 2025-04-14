import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from copy import deepcopy
from roco.dataset import ROCO

class ROCODataLoader(DataLoader):
    def __init__(self, data, tokenizer, args, split='train', shuffle=True):
        self.data = data
        self.tokenizer = tokenizer
        self.args = args
        self.batch_size = args.batch_size
        self.split = split
        self.shuffle = shuffle
        self.drop_last = True if split == 'train' else False

        # Image transformations based on encoder type
        if args.ve_name == "resnet":
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ]) if split == 'train' else transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

        elif args.ve_name == "efficientnet":
            self.transform = transforms.Compose([
                transforms.Resize((args.input_size, args.input_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        else:
            raise ValueError(f"Unsupported backbone: {args.ve_name}")

        # Dataset
        self.dataset = ROCO(
            args=self.args,
            data=self.data,
            tokenizer=self.tokenizer,
            split=self.split,
            transform=self.transform
        )

    @staticmethod
    def custom_collate(batch):
        image_ids,images, tokens,target_tokens,captions = zip(*batch)

        # Stack images into a batch
        images = torch.stack(images)

        # Pad sequences of tokens to the same length
        desc_tokens = torch.nn.utils.rnn.pad_sequence(tokens, batch_first=True, padding_value=0)
        target_tokens = torch.nn.utils.rnn.pad_sequence(target_tokens, batch_first=True, padding_value=0)

        # Return a tuple of batches
        return image_ids, images, desc_tokens, target_tokens,captions

        