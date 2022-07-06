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
    device = "mps"
    execution_device = torch.device(device)

    # Define and create Training Dataset and Test Dataset
    '''training_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/trainB", "datasets/my_simpson_dataset/trainA")
    test_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/testB", "datasets/my_simpson_dataset/testA")'''
    
    training_data = FaceToSimpsonDataset("datasets/tiny_dataset/trainB", "datasets/tiny_dataset/trainA")
    test_data = FaceToSimpsonDataset("datasets/tiny_dataset/testB", "datasets/tiny_dataset/testA", apply_transforms=False)

    # Create Data Loaders
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=4, shuffle=True)

    # Create Model
    cycleGAN = CycleGAN(training_phase=True, device=execution_device, save_dir="checkpoints",
                        lambda_cycle_loss=10.0, init_gain=0.02)

    print("Starting training...")

    print(f"Device used: {execution_device}")

    epochs = 5

    for epoch in range(epochs):
        for i, input_data in enumerate(train_dataloader):
            x_images, y_images = input_data
            cycleGAN.set_input_tensors(x_images, y_images)
            print("Forward pass...")
            cycleGAN.train_step()

            print(f"Epoch: {epoch + 1}, Step: {i + 1}/{len(train_dataloader)}")

        if epoch % 3 == 0 and epoch > 0:
            print("Saving model...")
            cycleGAN.save_checkpoints(epoch)