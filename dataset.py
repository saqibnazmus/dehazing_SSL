import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Resize, functional as F

class HazeLowlightDataset(Dataset):
    def __init__(self, hazy_dir, low_light_dir, tx_map_dir, a_map_dir, l_map_dir, transform=None, resize_to=(256, 256)):
        """
        Dataset for haze and low-light images with corresponding transmission maps, atmospheric light maps, and illumination maps.

        Args:
            hazy_dir (str): Path to the folder containing hazy images (.JPG/.PNG).
            low_light_dir (str): Path to the folder containing low-light images (.JPG/.PNG).
            tx_map_dir (str): Path to the folder containing transmission maps (.npy).
            a_map_dir (str): Path to the folder containing atmospheric light maps (.npy).
            l_map_dir (str): Path to the folder containing illumination maps (.npy).
            transform (callable, optional): Transform to apply to images.
            resize_to (tuple): Tuple specifying the (height, width) to resize images and maps.
        """
        self.hazy_dir = hazy_dir
        self.low_light_dir = low_light_dir
        self.tx_map_dir = tx_map_dir
        self.a_map_dir = a_map_dir
        self.l_map_dir = l_map_dir
        self.transform = transform
        self.resize_to = resize_to

        self.hazy_images = sorted(os.listdir(hazy_dir))
        self.low_light_images = sorted(os.listdir(low_light_dir))
        self.t_maps = sorted(os.listdir(tx_map_dir))
        self.a_maps = sorted(os.listdir(a_map_dir))
        self.l_maps = sorted(os.listdir(l_map_dir))

        assert len(self.hazy_images) == len(self.low_light_images) == len(self.t_maps) == len(self.a_maps) == len(self.l_maps), \
            "Mismatch in the number of files between directories."

        self.resizer = Resize(self.resize_to)

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        hazy_image = Image.open(hazy_image_path).convert("RGB")
        hazy_image = self.resizer(hazy_image)

        low_light_image_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        low_light_image = Image.open(low_light_image_path).convert("RGB")
        low_light_image = self.resizer(low_light_image)

        t_map = self._load_and_resize_map(self.tx_map_dir, self.t_maps[idx])
        a_map = self._load_and_resize_map(self.a_map_dir, self.a_maps[idx])
        l_map = self._load_and_resize_map(self.l_map_dir, self.l_maps[idx])

        if self.transform:
            hazy_image = self.transform(hazy_image)
            low_light_image = self.transform(low_light_image)

        return hazy_image, low_light_image, t_map, a_map, l_map

    def _load_and_resize_map(self, map_dir, map_filename):
        map_path = os.path.join(map_dir, map_filename)
        map_data = np.load(map_path)
        map_tensor = torch.tensor(map_data, dtype=torch.float32)
        return map_tensor

class HazyLowlightDataset_1(Dataset):
    def __init__(self, hazy_dir, low_light_dir, transform=None, resize_to=(256, 256)):
        self.hazy_dir = hazy_dir
        self.low_light_dir = low_light_dir
        self.transform = transform
        self.resize_to = resize_to

        self.hazy_images = sorted([f for f in os.listdir(hazy_dir) if f.endswith(('.jpg', '.png'))])
        self.low_light_images = sorted([f for f in os.listdir(low_light_dir) if f.endswith(('.jpg', '.png'))])

        self.resizer = Resize(self.resize_to)

    def __len__(self):
        return len(self.hazy_images)

    def __getitem__(self, idx):
        hazy_image_path = os.path.join(self.hazy_dir, self.hazy_images[idx])
        hazy_image = Image.open(hazy_image_path).convert("RGB")
        hazy_image = self.resizer(hazy_image)
        hazy_tensor = F.to_tensor(hazy_image)

        t_map = self._generate_transmission_map(hazy_tensor)
        a_map = self._estimate_atmospheric_light(hazy_tensor)

        low_light_image_path = os.path.join(self.low_light_dir, self.low_light_images[idx])
        low_light_image = Image.open(low_light_image_path).convert("RGB")
        low_light_image = self.resizer(low_light_image)
        low_light_tensor = F.to_tensor(low_light_image)

        l_map = self._estimate_illumination_map(low_light_tensor)

        return hazy_tensor, low_light_tensor, t_map, a_map, l_map

    def _generate_transmission_map(self, hazy_tensor):
        t_map = hazy_tensor.clone()
        t_map = t_map / t_map.max()
        return t_map

    def _estimate_atmospheric_light(self, hazy_tensor):
        a_map = hazy_tensor.mean(dim=(1, 2), keepdim=True)
        a_map = a_map.repeat(1, self.resize_to[0], self.resize_to[1])
        return a_map

    def _estimate_illumination_map(self, low_light_tensor):
        l_map = low_light_tensor.clone()
        l_map = l_map / l_map.max()
        return l_map
