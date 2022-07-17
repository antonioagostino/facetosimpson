from re import A
from data_interfaces.unpaired_interfaces.face_to_simpson_interface import FaceToSimpsonDataset
from torch.utils.data import DataLoader
from models.CycleGAN import CycleGAN
from utils import bcolors
import torch

if __name__ == "__main__":

    # If the training has to start from a checkpoint
    restore_training = True
    epoch_to_restore_from = 100

    # Set Torch Device
    device = "cuda"
    execution_device = torch.device(device)

    # Define and create Training Dataset 
    training_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/trainB", "datasets/my_simpson_dataset/trainA")
    
    # Define a dummy dataset for testing
    dummy_training_data = FaceToSimpsonDataset("datasets/tiny_dataset/trainB", "datasets/tiny_dataset/trainA")

    batch_size = 1

    # Create Data Loader
    train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)


    # Create Model
    cycleGAN = CycleGAN(training_phase=True, device=execution_device, save_dir="checkpoints",
                        generated_images_dir="", lambda_cycle_loss=10.0, init_gain=0.02)

    print(f"Strart training with device: {bcolors.CYAN}{execution_device}{bcolors.WHITE}")

    epochs = 150
    num_batches = len(train_dataloader)
    epoch_to_start_from = 1

    # Parameters of the training progress bar
    progress_bar_lenght = 20
    progress_bar_step = progress_bar_lenght / num_batches
    progress_bar_counter = 0

    # If the training have to start from a checkpoint
    if restore_training:
        print(f"{bcolors.CYAN}Restoring training from epoch {epoch_to_restore_from}{bcolors.WHITE}")
        cycleGAN.load_checkpoints(epoch=epoch_to_restore_from, restore_training=True)
        epoch_to_start_from = epoch_to_restore_from + 1

    for epoch in range(epoch_to_start_from, epochs + 1):

        print(f"{bcolors.CYAN}Epoch: {bcolors.GREEN}{epoch}{bcolors.CYAN}/{epochs}{bcolors.WHITE}")
        print("Steps: [", end="", flush=True)
        for i, input_data in enumerate(train_dataloader):
            x_images, y_images = input_data
            cycleGAN.set_input_tensors(x_images, y_images)
            cycleGAN.train_step()

            # Update progress bar
            progress_bar_counter += progress_bar_step
            if progress_bar_counter >= 1:
                progress_bar_counter = int(progress_bar_counter)
                for i in range(progress_bar_counter):
                    print("=", end="", flush=True)

                progress_bar_counter = 0

        print("]")
        cycleGAN.print_losses()

    print(f"{bcolors.WHITE}Training completed")