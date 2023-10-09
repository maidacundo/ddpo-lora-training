from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from pathlib import Path
import os
from utils import create_dataset

import numpy as np

class SamplingDataset(Dataset):
    """
    A dataset to prepare the instance and class images with the prompts for fine-tuning the model.
    It pre-processes the images and the tokenizes prompts.
    """

    def __init__(
        self,
        instance_data_root,
        prompt_fn,
        min_size=512,
        max_size=768,
        resize=True,
        normalize=True,
    ):  
        self.prompt_fn = prompt_fn
        self.resize = resize
        self.min_size = min_size
        self.max_size = max_size

        instance_data_root = Path(instance_data_root)

        if not Path(instance_data_root).exists():
            raise ValueError("Instance images root doesn't exists.")
        images_path = os.path.join(instance_data_root, "images")
        labels_path = os.path.join(instance_data_root, "labels")

        if not Path(images_path).exists():
            raise ValueError("Instance images root doesn't exists.")
        if not Path(labels_path).exists():
            raise ValueError("Instance labels root doesn't exists.")

        self.images, self.masks, self.labels = create_dataset(images_path, labels_path)
        self.masks = self.masks * 255

        self.num_instance_images = len(self.images)

        self._length = len(self.images)

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=(self.min_size, self.min_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ) # the image size is as maximum 768x512
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5])
                if normalize
                else transforms.Lambda(lambda x: x),
            ]
        )
        self.mask_transforms = transforms.Compose(
            [
                transforms.Resize(
                    size=(self.min_size, self.min_size),
                    interpolation=transforms.InterpolationMode.BILINEAR,
                ) # the image size is as maximum 768x512
                if resize
                else transforms.Lambda(lambda x: x),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return self._length

    def __getitem__(self, index):

        example = {}

        image = Image.fromarray(self.images[index % self.num_instance_images])
        mask = Image.fromarray(self.masks[index % self.num_instance_images])

        if not image.mode == "RGB":
            image = image.convert("RGB")

        
        example["image"] = self.image_transforms(image)
        example["mask"] = self.mask_transforms(mask)
        example["prompt"] = self.prompt_fn()

        return example
    