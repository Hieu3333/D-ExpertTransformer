from torch.utils.data import DataLoader
from torchvision import transforms
import torch
from modules.dataset import DeepEyeNet

class DENDataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        # Ensure the required arguments are provided
        assert hasattr(args, 'batch_size'), "args.batch_size is required"
        assert hasattr(args, 'num_workers'), "args.num_workers is required"
        assert hasattr(args, 've_name'), "args.ve_name is required"
        
        self.args = args
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.drop_last = True if split == 'train' or args.use_learnable_tokens else False

        # Define Image Transformations based on backbone model (ve_name)
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
            # Dynamically setting the size based on input size from the args (if specified)
            self.transform = transforms.Compose([
                transforms.Resize((356, 356)),  # Resizing to match input size
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
        elif args.ve_name == "densenet":
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
        else:
            raise ValueError(f"Unsupported backbone: {args.ve_name}")

        # Initialize Dataset
        self.dataset = DeepEyeNet(
            args=args,
            tokenizer=tokenizer,
            split=split,
            transform=self.transform
        )

        # Initialize DataLoader with the custom collate function
        super().__init__(
            dataset=self.dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            drop_last=self.drop_last,
            collate_fn=self.custom_collate  # Pass custom collate function
        )

    @staticmethod
    def custom_collate(batch):
        """Custom collate function to handle batches of images and tokens."""
        image_ids, images, desc_tokens, target_tokens, keyword_tokens, clinical_descs = zip(*batch)

        # Stack images into a batch
        images = torch.stack(images)

        # Pad sequences of tokens to the same length
        desc_tokens = torch.nn.utils.rnn.pad_sequence(desc_tokens, batch_first=True, padding_value=0)
        target_tokens = torch.nn.utils.rnn.pad_sequence(target_tokens, batch_first=True, padding_value=0)
        keyword_tokens = torch.nn.utils.rnn.pad_sequence(keyword_tokens, batch_first=True, padding_value=0)

        # Return a tuple of batches
        return image_ids, images, desc_tokens, target_tokens, keyword_tokens, clinical_descs  # clinical_descs remains a list of strings

    # def set_mask_prob(self,prob):
    #     self.dataset.set_masking_probability(prob)