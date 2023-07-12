import os
import glob
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np


def load_dataset(root_dir: str, image_size: tuple, batch_size: int, dataset_size: float = 1.0,
                 shuffle: bool = False, num_workers: int = 0, img_format: str = 'L'):
    dataset = CustomDataset(root_dir, image_size, img_format)
    dataset_size = int(len(dataset) * dataset_size)
    rest_of_dataset_size = int(len(dataset) - dataset_size)
    dataset, _ = random_split(dataset, [dataset_size, rest_of_dataset_size])
    data_loader = DataLoader(
        dataset, batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return data_loader


class CustomDataset(Dataset):
    def __init__(self, root_dir, image_size, img_format):
        self.files = sorted(
            glob.glob(os.path.join(root_dir, '*.[jp][pn]g')) + glob.glob(os.path.join(root_dir, '*.tif')))

        self.transform = transforms.Compose([
            transforms.Resize(image_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
            # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        ])
        self._format = img_format

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        img_path = self.files[idx]
        image = Image.open(img_path).convert(self._format)
        image_tensor = self.transform(image)[None, ...]
        image = np.array(image)
        return image_tensor, image, filename
