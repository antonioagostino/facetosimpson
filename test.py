from itertools import cycle
from data_interfaces.unpaired_interfaces.face_to_simpson_interface import FaceToSimpsonDataset
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
import torchvision.transforms as T
from models.CycleGAN import CycleGAN
import os

if __name__ == "__main__":

    # Set Torch Device
    device = "cpu"
    execution_device = torch.device(device)

    # Define and create Test Dataset
    '''test_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/testB", "datasets/my_simpson_dataset/testA", apply_transforms=False)'''
    
    test_data = FaceToSimpsonDataset("datasets/tiny_dataset/testB", "datasets/tiny_dataset/testA", apply_transforms=False)

    # Create Data Loader
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    # Create Model
    cycleGAN = CycleGAN(training_phase=True, device=execution_device, save_dir="checkpoints",
                        lambda_cycle_loss=10.0, init_gain=0.02)

    print(f"Device used: {execution_device}")

    print("Testing...")
    cycleGAN_test = CycleGAN(training_phase=True, device=execution_device, save_dir="checkpoints",
                        lambda_cycle_loss=10.0, init_gain=0.02)

    cycleGAN_test.load_checkpoints(3)

    for i, input_data in enumerate(test_dataloader):
        x_images, y_images = input_data
        cycleGAN.set_input_tensors(x_images, y_images)
        cycleGAN.test_step()