from data_interfaces.unpaired_interfaces.face_to_simpson_interface import FaceToSimpsonDataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from models.CycleGAN import CycleGAN
from utils import bcolors
from scores.fid_scores import cal_fid as fid_score
import data_ios
import shutil
import torch
import os

if __name__ == "__main__":

    restore_training = False
    generated_images_dir = "generated"
    generated_filenames_file_path = "gt_list.txt"
    generated_filenames_file = open(generated_filenames_file_path, "w")

    # Set Torch Device
    device = "cpu"
    execution_device = torch.device(device)

    # Define and create Training Dataset and Test Dataset
    '''training_data = FaceToSimpsonDataset("datasets/my_simpson_dataset/trainB", "datasets/my_simpson_dataset/trainA")'''
    
    training_data = FaceToSimpsonDataset("datasets/tiny_dataset/trainB", "datasets/tiny_dataset/trainA")

    training_set_portion = 0.85
    training_set_size = int(training_set_portion * len(training_data))

    training_data, validation_data = torch.utils.data.random_split(training_data, [training_set_size, len(training_data) - training_set_size])

    # Create Data Loaders
    train_dataloader = DataLoader(training_data, batch_size=4, shuffle=True)
    val_dataloader = DataLoader(validation_data, batch_size=1, shuffle=True)


    # Create Model
    cycleGAN = CycleGAN(training_phase=True, device=execution_device, save_dir="checkpoints",
                        generated_images_dir="generated", lambda_cycle_loss=10.0, init_gain=0.02)

    print(f"Strart training with device: {bcolors.CYAN}{execution_device}{bcolors.WHITE}")

    epochs = 5
    num_batches = len(train_dataloader)
    epoch_to_start_from = 1
    progress_bar_lenght = 20
    progress_bar_step = progress_bar_lenght / num_batches
    progress_bar_counter = 0
    validation_set_dir = "validation_set"
    validation_filenames_filepath = "predlist.txt"
    validation_filenames_file = open(validation_filenames_filepath, "w")

    if restore_training:
        epoch_to_restore_from = 3
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

            progress_bar_counter += progress_bar_step
            if progress_bar_counter >= 1:
                for i in range(progress_bar_lenght):
                    print("=", end="", flush=True)

                progress_bar_counter = 0

        print("]")
        cycleGAN.print_losses()

        if epoch % 3 == 0 and epoch > 0:
            print(f"{bcolors.WHITE}Saving checkpoints")
            cycleGAN.save_checkpoints(epoch)

            print(f"{bcolors.WHITE}---------------")
            print("Validation step:")
            cycleGAN_val = CycleGAN(training_phase=False, device=execution_device, save_dir="checkpoints",
                                    generated_images_dir="generated", generated_filenames_file=generated_filenames_file)
            cycleGAN_val.load_checkpoints(epoch)
            os.makedirs(validation_set_dir, exist_ok=True)
            os.makedirs(generated_images_dir, exist_ok=True)

            for val_idx, val_data in enumerate(val_dataloader):
                x_image, y_image = val_data
                image_filename = f"{val_idx}.png"
                export_path = os.path.join(validation_set_dir, image_filename)
                save_image(y_image, export_path)
                validation_filenames_file.write(export_path + "\n")

                cycleGAN_val.set_input_tensors(x_image, y_image)
                cycleGAN_val.validation_step()

            
            validation_filenames_file.close()
            generated_filenames_file.close()
            real_data_generator = data_ios.data_prepare_fid_is(validation_filenames_filepath, 1, 299, False)
            fake_data_generator = data_ios.data_prepare_fid_is(generated_filenames_file_path, 1, 299, False)
            dims = 2048
            final_score = fid_score(real_data_generator, fake_data_generator, dims, False)
            print(f"{bcolors.CYAN}FID score: {bcolors.GREEN}{final_score}{bcolors.WHITE}")
            os.remove(validation_filenames_filepath)
            os.remove(generated_filenames_file_path)
            shutil.rmtree(validation_set_dir)
            shutil.rmtree(generated_images_dir)