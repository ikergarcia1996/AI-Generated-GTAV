from torch.utils.data import Dataset
from web_dataset import actions_to_one_hot
import torch


class ImageDataset(Dataset):
    def __init__(
        self,
        split: str,
        return_actions: bool = False,
    ):
        self.return_actions = return_actions
        self.split = split

        # Create the sequence of 5 images interpolating from blue to red
        blue = torch.tensor([0.0, 0.0, 1.0])
        red = torch.tensor([1.0, 0.0, 0.0])

        # Generate 5 interpolated colors
        self.images_blue_red = []
        for t in torch.linspace(0, 1, 5):
            color = (1 - t) * blue + t * red
            # Create image of size [3, 360, 640] filled with the interpolated color
            img = color.view(3, 1, 1).expand(3, 360, 640)
            self.images_blue_red.append(img)

        # Stack images into sequence [5, 3, 360, 640]
        self.sequence_blue_red = torch.stack(self.images_blue_red)

        # Create the sequence of 5 images interpolating from blue to green

        green = torch.tensor([0.0, 1.0, 0.0])
        green = green.view(3, 1, 1).expand(3, 360, 640)
        self.sequence_blue_green = self.sequence_blue_red.clone()
        # Repalce last image with green
        self.sequence_blue_green[-1] = green

    def __len__(self):
        return 10000000 if self.split == "train" else 10

    def __iter__(self):
        for _ in range(self.__len__()):
            if not self.return_actions:
                yield {"video": self.sequence_blue_red}
            else:
                actions = torch.randint(
                    0, 2, (self.sequence_blue_red.shape[0],), dtype=torch.long
                )
                actions[:-1] = -1
                last_action = actions[-1]
                actions = actions_to_one_hot(list(actions))
                if last_action == 0:
                    video = self.sequence_blue_red
                else:
                    video = self.sequence_blue_green
                yield {"video": video, "actions": actions}

    def __getitem__(self, index):
        if not self.return_actions:
            return {"video": self.sequence_blue_red}
        else:
            actions = torch.randint(
                0, 2, (self.sequence_blue_red.shape[0],), dtype=torch.long
            )
            actions[:-1] = -1
            last_action = actions[-1]
            actions = actions_to_one_hot(list(actions))
            if last_action == 0:
                video = self.sequence_blue_red
            else:
                video = self.sequence_blue_green
            return {"video": video, "actions": actions}
