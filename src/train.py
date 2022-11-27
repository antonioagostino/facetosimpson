from data_interfaces.unpaired_interfaces.face_to_simpson_interface import FaceToSimpsonDataset
from torch.utils.data import DataLoader
from models.CycleGAN import CycleGAN
from utils import bcolors
from scores.fid_scores import cal_fid as fid_score
import data_ios
import torch
import os
import shutil

if __name__ == "__main__":

    breakpoint()

    # If the training has to start from a checkpoint
    restore_training = False
    epoch_to_restore_from = 0

    # Used to find the best model
    best_fid = float("inf")

    # Where to save images generated by the generator (G: X -> Y)
    generated_images_dir = "generated"

    # A file containing the paths of all the images generated
    generated_filenames_file_path = "gtlist.txt"

    # A file containing the paths of all test set's images
    test_filenames_filepath = "predlist.txt"

    # Set Torch Device
    device = "cpu"
    execution_device = torch.device(device)

    # Define and create Training Dataset 
    training_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/trainB", "datasets/my_simpson_dataset/trainA")
    
    # Define a dummy dataset for testing
    dummy_training_data = FaceToSimpsonDataset("datasets/tiny_dataset/trainB", "datasets/tiny_dataset/trainA")

    # Define and create Test Dataset
    test_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/testB", "datasets/my_simpson_dataset/testA", apply_transforms=False)

    batch_size = 1

    # Create Training Data Loader
    train_dataloader = DataLoader(dummy_training_data, batch_size=batch_size, shuffle=True)

    # Create Test Data Loader
    test_dataloader = DataLoader(test_data, batch_size=1, shuffle=True)

    # Create Model
    cycleGAN = CycleGAN(training_phase=True, device=execution_device, save_dir="checkpoints",
                        generated_images_dir="", lambda_cycle_loss=10.0, init_gain=0.02)

    print(f"Strart training with device: {bcolors.CYAN}{execution_device}{bcolors.WHITE}")

    epochs = 5
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

        print(f"{bcolors.WHITE}-----------------------------------------------------")

        print(f"{bcolors.CYAN}Saving checkpoints...{bcolors.WHITE}")

        cycleGAN.save_checkpoints(best_model=False, epoch=epoch)

        os.makedirs(generated_images_dir, exist_ok=True)

        generated_filenames_file = open(generated_filenames_file_path, "w")

        cycleGAN_test = CycleGAN(training_phase=False, device=execution_device, save_dir="checkpoints", generated_images_dir=generated_images_dir,
                            generated_filenames_file=generated_filenames_file, lambda_cycle_loss=10.0, init_gain=0.02)

        print(f"FID evaluation started...")

        cycleGAN_test.load_checkpoints(best_model=False, restore_training=False)

        for i, input_data in enumerate(test_dataloader):
            x_images, y_images = input_data
            cycleGAN_test.set_input_tensors(x_images, y_images)
            cycleGAN_test.test_step()

        generated_filenames_file.close()

        real_data_generator = data_ios.data_prepare_fid_is(test_filenames_filepath, 1, 299, False)
        fake_data_generator = data_ios.data_prepare_fid_is(generated_filenames_file_path, 1, 299, False)
        dims = 2048

        final_score = fid_score(real_data_generator, fake_data_generator, dims, False)
        
        print(f"{bcolors.CYAN}FID score: {bcolors.GREEN}{final_score}{bcolors.WHITE}")
        
        os.remove(generated_filenames_file_path)
        shutil.rmtree(generated_images_dir)

        if final_score < best_fid:
            best_fid = final_score
            cycleGAN.save_checkpoints(best_model=True, epoch=epoch)
            print(f"{bcolors.CYAN}New best FID score: {bcolors.GREEN}{best_fid}{bcolors.WHITE}")

        print(f"{bcolors.WHITE}-----------------------------------------------------")

        cycleGAN.update_learning_rate()

        print(f"{bcolors.WHITE}-----------------------------------------------------")

    print(f"{bcolors.WHITE}Training completed")