from torch.utils.data import Dataset
import torchvision.transforms as T
import os
import random
from PIL import Image

class FaceToSimpsonDataset(Dataset):
    """
    A class that implements an interface for a Dataset of two domains: X and Y.
    X contains images of human faces, Y contains images of Simpson's faces.
    """
    def __init__(self, x_dir: str, y_dir: str, apply_transforms: bool = True):
        """
            Initializes the dataset.
            :param x_dir: path to the directory containing the images of the domain X
            :param y_dir: path to the directory containing the images of the domain Y
            :param apply_transforms: if True, applies the training transforms to the images, 
                otherwise applies the test transforms
        """
        self.x_dir = x_dir
        self.y_dir = y_dir
        self.x_filenames = [name for name in os.listdir(self.x_dir) if os.path.isfile(os.path.join(self.x_dir, name))]
        self.y_filenames = [name for name in os.listdir(self.y_dir) if os.path.isfile(os.path.join(self.y_dir, name))]
        self.x_size = len(self.x_filenames)
        self.y_size = len(self.y_filenames)
        self.apply_transforms = apply_transforms
        self.transform = T.Compose([
            T.ToTensor(),
            T.Resize([256, 256]),
            T.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            T.RandomRotation(10),
            T.RandomHorizontalFlip()
        ])

        self.test_transform = T.Compose([
            T.ToTensor(),
            T.Resize([256, 256])
        ])

    def __len__(self):
        # we are sure that the number of images in the two directories are the same
        return self.x_size

    def __getitem__(self, idx):
        x_img_path = os.path.join(self.x_dir, self.x_filenames[idx])
        x_image = Image.open(x_img_path).convert('RGB')
        y_img_path = os.path.join(self.y_dir, self.y_filenames[random.randint(0, self.y_size - 1)])
        y_image = Image.open(y_img_path).convert('RGB')
        if self.apply_transforms:
            x_image = self.transform(x_image)
            y_image = self.transform(y_image)
        else:
            x_image = self.test_transform(x_image)
            y_image = self.test_transform(y_image)

        return x_image, y_image