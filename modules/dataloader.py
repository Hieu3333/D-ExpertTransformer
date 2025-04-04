from torch.utils.data import DataLoader
from torchvision import transforms
from modules.dataset import DeepEyeNet
import torch

class DENDataLoader(DataLoader):
    def __init__(self, args, tokenizer, keywords_vocab_set, split, shuffle):
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.drop_last = True if split == 'train' else False

        # Define Image Transformations
        if args.ve_name == "resnet":
            if split == 'train':
                self.transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.RandomCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406),
                                         (0.229, 0.224, 0.225))
                ])

        elif args.ve_name == "efficientnet":
            self.transform = transforms.Compose([
                transforms.Resize((356, 356)),  # Resize to match input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])

        # Initialize Dataset
        if args.decoder == "custom":
            self.dataset = DeepEyeNet(
            args=args,
            tokenizer=tokenizer,
            keywords_list=keywords_vocab_set,  # set of keywords, non-duplicated
            split=split,
            transform=self.transform
        )
        if args.decoder == "gpt2":
            self.dataset = DeepEyeNet_GPT(
            args=args,
            tokenizer=tokenizer,
            keywords_list=keywords_vocab_set,  # set of keywords, non-duplicated
            split=split,
            transform=self.transform
        )

        # Custom collate function to handle raw strings
        def custom_collate(batch):
            image_ids, images, desc_tokens, target_tokens, keyword_tokens, clinical_descs = zip(*batch)

            images = torch.stack(images)  # Stack images into a batch
            desc_tokens = torch.nn.utils.rnn.pad_sequence(desc_tokens, batch_first=True, padding_value=0)
            target_tokens = torch.nn.utils.rnn.pad_sequence(target_tokens, batch_first=True, padding_value=0)
            keyword_tokens = torch.nn.utils.rnn.pad_sequence(keyword_tokens, batch_first=True, padding_value=0)

            return image_ids, images, desc_tokens, target_tokens, keyword_tokens, clinical_descs  # clinical_descs remains a list of strings

        # Initialize DataLoader with the custom collate function
        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=custom_collate  # Pass custom collate function
        )
