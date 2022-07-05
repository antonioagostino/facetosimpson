from data_interfaces.unpaired_interfaces.face_to_simpson_interface import FaceToSimpsonDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(np.squeeze(npimg, axis=0), (1, 2, 0)))
    plt.show()

if __name__ == "__main__":
    training_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/trainB", "datasets/my_simpson_dataset/trainA")
    test_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/testB", "datasets/my_simpson_dataset/testA")
    train_dataloader = DataLoader(training_data, batch_size=1, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    x_img, y_img = iter(train_dataloader).next()
    imshow(y_img)
