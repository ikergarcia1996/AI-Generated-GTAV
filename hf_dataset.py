from typing import List

import torch
from datasets import load_dataset
from einops import rearrange
from torch.utils.data import Dataset
from torchvision import transforms


def action_to_one_hot(action: int) -> torch.tensor:
    # ... existing one_hot conversion code ...
    actions_tensor = torch.tensor([0, 0, 0, 0, action + 1], dtype=torch.long)
    return torch.nn.functional.one_hot(actions_tensor, num_classes=10)


def actions_to_one_hot(actions: List[int]) -> torch.tensor:
    """
    Converts actions to one-hot encoded vectors using torch.scatter_.
    Handles -1 values by creating a zero vector (no action).

    Args:
        actions (List[int]): Actions to convert, can contain -1 for no action.

    Returns:
        torch.tensor: One-hot encoded actions of shape (len(actions), 9).
    """
    actions_tensor = torch.tensor(actions)
    actions_tensor += 1  # to handle -1 values
    one_hot = torch.zeros(len(actions), 10, dtype=torch.long)
    one_hot.scatter_(1, actions_tensor.unsqueeze(1), 1)
    return one_hot


class SplitImages(object):
    """
    Splits a sequence image file into 5 images
    """

    def __call__(self, image: torch.tensor) -> torch.tensor:
        """
        Applies the transformation to the sequence of images.

        Args:
            image (np.array): Sequence of images. Size [3, 270, 2400]

        Returns:
            torch.tensor: Transformed sequence of images. Size (5, 270, 480, 3)
        """
        return rearrange(image, "c h (n w) -> n c h w", n=5, c=3, h=270, w=480)


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
