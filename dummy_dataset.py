
from torch.utils.data import IterableDataset

import torch

class ImageDataset(IterableDataset):
    def __init__(
        self,
        split: str,
        return_actions: bool = False,
    ):
        if return_actions:
            raise NotImplementedError("return_actions is not supported in this dataset")
        self.split = split
        
        # Create the sequence of 5 images interpolating from blue to red
        blue = torch.tensor([0.0, 0.0, 1.0])
        red = torch.tensor([1.0, 0.0, 0.0])
        
        # Generate 5 interpolated colors
        self.images = []
        for t in torch.linspace(0, 1, 5):
            color = (1 - t) * blue + t * red
            # Create image of size [3, 360, 640] filled with the interpolated color
            img = color.view(3, 1, 1).expand(3, 360, 640)
            self.images.append(img)
        
        # Stack images into sequence [5, 3, 360, 640]
        self.sequence = torch.stack(self.images)

    def __len__(self):
        return 10000000 if self.split == "train" else 10
    
    def __iter__(self):
        for _ in range(self.__len__()):
            yield {"video": self.sequence}
          