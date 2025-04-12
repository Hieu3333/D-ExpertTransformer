from torch.utils.data import DataLoader
from torchvision import transforms
from iu_xray.dataset import IUXray
import torch

class IUXrayDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
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
                    transforms.CenterCrop(224),  # Replaces RandomResizedCrop with deterministic crop
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
        # Initialize Dataset
        self.dataset = IUXray(
            args=args,
            tokenizer=tokenizer,
            split=split,
            transform=self.transform
        )

        # Custom collate function to handle raw strings
        def custom_collate(batch):
            image_ids, images, report_tokens, target_tokens, cleaned_report = zip(*batch)

            images = torch.stack(images)  # Stack images into a batch
            report_tokens = torch.nn.utils.rnn.pad_sequence(report_tokens, batch_first=True, padding_value=0)
            target_tokens = torch.nn.utils.rnn.pad_sequence(target_tokens, batch_first=True, padding_value=0)

            return image_ids, images, report_tokens, target_tokens, cleaned_report  # Return image_ids, images, and tokenized data

        # Initialize DataLoader with the custom collate function
        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=custom_collate  # Pass custom collate function
        )
