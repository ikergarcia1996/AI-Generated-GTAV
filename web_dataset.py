import glob
import os
from typing import List

import torch
import torch.multiprocessing
import webdataset as wds
from einops import rearrange
from huggingface_hub import HfFileSystem, get_token, hf_hub_url
from torch.utils.data import IterableDataset
from torchvision import transforms


def count_examples(dataset_dir: str) -> int:
    return len(glob.glob(os.path.join(dataset_dir, "*.jpeg")))


def split_len(split: str) -> int:
    return {"train": 1270669, "validation": 4040, "test": 4588}[split]


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
    one_hot = torch.zeros(len(actions), 9,dtype=torch.long)
    mask = actions_tensor >= 0  # Changed from != -1 for clarity
    if mask.any():
        one_hot[torch.arange(len(actions))[mask], actions_tensor[mask]] = 1
    return one_hot.float()


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


class ImageDataset(IterableDataset):
    def __init__(
        self,
        split: str,
        return_actions: bool = False,
    ):
        """
        INIT
        Args:
            split (str): Split of the dataset. One of ["train", "validation", "test"]
            return_actions (bool): If True, return one-hot encoded actions
        """
        self.return_actions = return_actions
        self.split = split

        # Define split patterns with correct paths
        splits = {
            "train": "**/train/*.tar",
            "validation": "dev/00000.tar",  # Updated path
            "test": "**/test/**/*.tar",  # Updated path to specifically look for tar files
        }

        # Set up HuggingFace filesystem and get URLs
        fs = HfFileSystem()
        pattern = f"hf://datasets/Iker/GTAV-Driving-Dataset/{splits[split]}"
        files = [fs.resolve_path(path) for path in fs.glob(pattern)]

        if not files:
            raise ValueError(
                f"No files found for split '{split}' with pattern {pattern}"
            )

        urls = [
            hf_hub_url(file.repo_id, file.path_in_repo, repo_type="dataset")
            for file in files
        ]

        # Join URLs with double colon and add curl command

        urls = (
            f"pipe:curl -s -L --retry 3 --retry-delay 1 --retry-all-errors "
            f"-H 'Authorization:Bearer {get_token()}' {'::'.join(urls)}"
        )

        transform = transforms.Compose(
            [transforms.ToTensor(), SplitImages(), transforms.Resize((360, 640))]
        )

        # Create WebDataset with proper image decoding
        self.dataset = (
            wds.WebDataset(
                urls,
                handler=wds.warn_and_continue,
                shardshuffle=True,
                nodesplitter=wds.shardlists.split_by_worker,
                empty_check=False,
                resampled=True,
            )
            .shuffle(1000)  # Add shuffle buffer
            .decode("pil")  # Decode as PIL Image
            .to_tuple("jpg", "cls", "json")
            .map(lambda x: (transform(x[0]), x[1], x[2]))  # Pass through all three values
        )

        print(f"Loaded dataset for {split} split with {len(files)} tar files")


    def __len__(self):
        """
        Returns the number of examples in the dataset.
        """
        return split_len(self.split)
    
    def __iter__(self):
        """
        Returns a sample from the dataset.
        """
        for img, cls, json_data in self.dataset:
            if self.return_actions:
                actions = actions_to_one_hot(json_data["actions_int"])
                yield {"video": img, "actions": actions}
            else:
                yield {"video": img}
