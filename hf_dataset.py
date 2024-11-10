
from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision import transforms

from web_dataset import SplitImages, action_to_one_hot


class ImageDataset(Dataset):
    def __init__(
        self,
        split: str,
        return_actions: bool = False,
    ):
        self.return_actions = return_actions
        self.split = split

        # Load dataset using Hugging Face datasets
        self.dataset = load_dataset("Iker/GTAV-Driving-Dataset", split=split)

        # Define transforms
        self.transform = transforms.Compose(
            [transforms.ToTensor(), SplitImages(), transforms.Resize((360, 640))]
        )

        print(f"Loaded dataset for {split} split with {len(self.dataset)} examples")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        sample = self.dataset[idx]

        # Convert PIL image to tensor and apply transforms
        img = self.transform(sample["jpg"])

        if self.return_actions:
            actions = action_to_one_hot(sample["label"])
            return {"video": img, "actions": actions}
        else:
            return {"video": img}

    def __iter__(self):
        for sample in self.dataset:
            # Convert PIL image to tensor and apply transforms
            img = self.transform(sample["image"])

            if self.return_actions:
                actions = action_to_one_hot(sample["label"])
                yield {"video": img, "actions": actions}
            else:
                yield {"video": img}
