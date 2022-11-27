from distutils.command import check
import torch.nn as nn
from torch.nn import init
import torch
import torchvision
from torch.optim import lr_scheduler
from torchvision.utils import save_image
import random
from .ResNetGenerator import ResNetGenerator
from .PatchGANDiscriminator import PatchGANDiscriminator
from losses.LeastSquaresGANLoss import LeastSquaresGANLoss
import os
import numpy as np
import matplotlib.pyplot as plt
from utils import bcolors

class CycleGAN(nn.Module):
    """
    Implementation of the CycleGAN model from the paper: https://arxiv.org/pdf/1703.10593.pdf.
    This implementation uses the default configuration:
        - The generator is a ResNet with 9 ResNet blocks.
        - The discriminator is a PatchGAN discriminator (from Pix2Pix).
        - As GAN objective, least-square GAN loss is used.
        - Identity loss is not used.
        - No dropout is used.
    """
    def __init__(self, training_phase: bool, device: torch.device, save_dir: str, 
                    generated_images_dir: str, generated_filenames_file=None,
                    lambda_cycle_loss: float = 10.0, init_gain: float = 0.02):
        """
        Parameters:
            training_phase (bool): Whether the model is in training phase or not.
            lambda_cycle_loss (float): Objective weight of the cycle loss.
        """
        super().__init__()

        self.device = device
        self.save_dir = save_dir
        self.generated_images_dir = generated_images_dir
        self.generated_filenames_file = generated_filenames_file
        self.image_export_counter = 0

        # Define G: X -> Y mapping function and F: Y -> X mapping function
        self.G = ResNetGenerator(num_resnet_blocks=9, bias=True, normaliz_layer=nn.InstanceNorm2d).to(self.device)
        self.F = ResNetGenerator(num_resnet_blocks=9, bias=True, normaliz_layer=nn.InstanceNorm2d).to(self.device)

        # A list of images generated during previuous steps
        self.G_generated_images = []
        self.F_generated_images = []
        self.G_buffer_size = 50
        self.F_buffer_size = 50

        if training_phase:
            
            self.init_gain = init_gain

            # Initialize G's weights
            self.G.apply(self.__initialize_weights)

            # Initialize F's weights
            self.F.apply(self.__initialize_weights)

            # Objective weight for the cycle loss
            self.lambda_cycle_loss = lambda_cycle_loss

            # Define D_X which aims to discriminate between real X images and F(Y) images
            self.D_X = PatchGANDiscriminator(bias=True, normaliz_layer=nn.InstanceNorm2d).to(self.device)

            # Initialize D_X's weights
            self.D_X.apply(self.__initialize_weights)

            # Define D_Y which aims to discriminate between real Y images and G(X) images
            self.D_Y = PatchGANDiscriminator(bias=True, normaliz_layer=nn.InstanceNorm2d).to(self.device)

            # Initialize D_Y's weights
            self.D_Y.apply(self.__initialize_weights)

            # Define loss functions
            self.adversarialLoss = LeastSquaresGANLoss(self.device).to(self.device)
            self.cycleLoss = nn.L1Loss()

            # Define optimizers
            self.optimizer_G = torch.optim.Adam([
                {'params': self.G.parameters()}, 
                {'params': self.F.parameters()}], 
                lr=0.0002, betas=(0.5, 0.999)
            
            )
            self.optimizer_D = torch.optim.Adam([
                {'params': self.D_Y.parameters()},
                {'params': self.D_X.parameters()}],
                lr=0.0002, betas=(0.5, 0.999)
            )

            self.gen_lr_scheduler = lr_scheduler.LinearLR(self.optimizer_G, 1.0, 0.0, 100)
            self.disc_lr_scheduler = lr_scheduler.LinearLR(self.optimizer_D, 1.0, 0.0, 100)

    def __initialize_weights(self, network: nn.Module):
        if hasattr(network, 'weight') and network.weight is not None:
            init.normal_(network.weight.data, 0.0, self.init_gain)
    
    def set_input_tensors(self, x_img: torch.Tensor, y_img: torch.Tensor):
        self.real_x = x_img.to(self.device)
        self.real_y = y_img.to(self.device)

    def update_learning_rate(self):
        old_lr = self.optimizer_G.param_groups[0]['lr']
        self.gen_lr_scheduler.step()
        self.disc_lr_scheduler.step()

        new_lr = self.optimizer_G.param_groups[0]['lr']
        print(f"{bcolors.CYAN}Learning Rate updated from {bcolors.GREEN}{old_lr}{bcolors.CYAN} to {bcolors.GREEN}{new_lr}{bcolors.WHITE}")
    
    def __get_generated_image(self, generator_buffer: list[torch.Tensor], generator_buffer_limit: int, images: list[torch.Tensor]):
        images_to_return = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if len(generator_buffer) < generator_buffer_limit:
                generator_buffer.append(image)
                images_to_return.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:
                    random_id = random.randint(0, generator_buffer_limit - 1)
                    tmp = generator_buffer[random_id].clone()
                    generator_buffer[random_id] = image
                    images_to_return.append(tmp)
                else:
                    images_to_return.append(image)
        
        return torch.cat(images_to_return, 0)
    
    def forward(self):
        self.fake_y = self.G(self.real_x)
        self.fake_x = self.F(self.real_y)
        self.reconstruction_x = self.F(self.fake_y)
        self.reconstruction_y = self.G(self.fake_x)

    
    def __backward_generators(self):
        # Adversarial Loss D_Y(G(X))
        self.loss_G = self.adversarialLoss(self.D_Y(self.fake_y), True)
        # Adversarial Loss D_X(F(Y))
        self.loss_F = self.adversarialLoss(self.D_X(self.fake_x), True)
        # Cycle Loss || F(G(X)) - X||
        self.loss_cycle_X = self.cycleLoss(self.reconstruction_x, self.real_x)
        # Cycle Loss || G(F(Y)) - Y||
        self.loss_cycle_Y = self.cycleLoss(self.reconstruction_y, self.real_y)
        
        # Full objective
        self.generators_loss = self.loss_G + self.loss_F + self.lambda_cycle_loss * (self.loss_cycle_X + self.loss_cycle_Y)
        self.generators_loss.backward()

    def print_losses(self):
        print(f"{bcolors.YELLOW}G Loss: {bcolors.RED}{self.loss_G.item():.4f}{bcolors.YELLOW}, F Loss: {bcolors.RED}{self.loss_F.item():.4f}{bcolors.YELLOW}, D_X Loss: {bcolors.RED}{self.loss_D_X.item():.4f}{bcolors.YELLOW}, D_Y Loss: {bcolors.RED}{self.loss_D_Y.item():.4f}{bcolors.YELLOW}, Cycle Loss X: {bcolors.RED}{self.loss_cycle_X.item():.4f}{bcolors.YELLOW}, Cycle Loss Y: {bcolors.RED}{self.loss_cycle_Y.item():.4f}")

    def export_losses(self):
        return f"G Loss: {self.loss_G.item():.4f}, F Loss: {self.loss_F.item():.4f}, D_X Loss: {self.loss_D_X.item():.4f}, D_Y Loss: {self.loss_D_Y.item():.4f}, Cycle Loss X: {self.loss_cycle_X.item():.4f}, Cycle Loss Y: {self.loss_cycle_Y.item():.4f}"
    
    def __backward_discriminators(self):

        # D_Y loss on real Y images
        D_Y_pred_on_real = self.D_Y(self.real_y)
        loss_D_Y_real = self.adversarialLoss(D_Y_pred_on_real, True)

        # D_Y loss on fake Y images
        fake_y_image = self.__get_generated_image(self.G_generated_images, self.G_buffer_size, self.fake_y)
        D_Y_pred_on_fake = self.D_Y(fake_y_image.detach())
        loss_D_Y_fake = self.adversarialLoss(D_Y_pred_on_fake, False)

        # D_Y whole loss (Least Squares) and backward pass
        self.loss_D_Y = (loss_D_Y_real + loss_D_Y_fake) * 0.5
        self.loss_D_Y.backward()

        # D_X loss on real X images
        D_X_pred_on_real = self.D_X(self.real_x)
        loss_D_X_real = self.adversarialLoss(D_X_pred_on_real, True)

        # D_X loss on fake X images
        fake_x_image = self.__get_generated_image(self.F_generated_images, self.F_buffer_size, self.fake_x)
        D_X_pred_on_fake = self.D_X(fake_x_image.detach())
        loss_D_X_fake = self.adversarialLoss(D_X_pred_on_fake, False)

        # D_X whole loss (Least Squares) and backward pass
        self.loss_D_X = (loss_D_X_real + loss_D_X_fake) * 0.5
        self.loss_D_X.backward()

    def __freeze_discriminators(self):
        for param in self.D_Y.parameters():
            param.requires_grad = False
        for param in self.D_X.parameters():
            param.requires_grad = False

    def __unfreeze_discriminators(self):
        for param in self.D_Y.parameters():
            param.requires_grad = True
        for param in self.D_X.parameters():
            param.requires_grad = True

    def train_step(self):
        self.forward()

        self.__freeze_discriminators()
        self.optimizer_G.zero_grad()
        self.__backward_generators()
        self.optimizer_G.step()

        self.__unfreeze_discriminators()
        self.optimizer_D.zero_grad()
        self.__backward_discriminators()
        self.optimizer_D.step()
    
    def __print_results(self, images):
        img_grid = torchvision.utils.make_grid(images)
        npimg = img_grid.numpy()
        plt.imshow(np.transpose(npimg, (1, 2, 0)))
        plt.show()

    def test_step(self):
        with torch.no_grad():
            self.forward()
            image_filename = f"{self.image_export_counter}.png"
            export_path = os.path.join(self.generated_images_dir, image_filename)
            save_image(self.fake_y, export_path)
            self.generated_filenames_file.write(export_path + "\n")
            self.image_export_counter += 1


    def save_checkpoints(self, best_model: bool, epoch: int):
        models_to_save = ["G", "F", "D_Y", "D_X"]
        for model in models_to_save:
            if best_model:
                save_filename = f"{model}_best.pt"
            else:
                save_filename = f"{model}_last.pt"
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, model)

            if self.device.type != 'cpu':
                torch.save(net.cpu().state_dict(), save_path)
                net.to(self.device)
            else:
                torch.save(net.state_dict(), save_path)

        optimizers = ['optimizer_G', 'optimizer_D']
        for optimizer in optimizers:
            if best_model:
                save_filename = f"{optimizer}_best.pt"
            else:
                save_filename = f"{optimizer}_last.pt"
            save_path = os.path.join(self.save_dir, save_filename)
            net = getattr(self, optimizer)
            torch.save(net.state_dict(), save_path)

        if best_model:
            checkpoint_info_file = open(os.path.join(self.save_dir, "checkpoint_info.txt"), "w")
            checkpoint_info_file.write(f"Best Model epoch: {epoch}\n")
            checkpoint_info_file.close()

    def load_checkpoints(self, best_model: bool, restore_training: bool = False):
        models_to_load = ["G", "F"]

        if restore_training:
            models_to_load.append("D_Y")
            models_to_load.append("D_X")
        
        for model in models_to_load:
            if best_model:
                load_filename = f"{model}_best.pt"
            else:
                load_filename = f"{model}_last.pt"
            load_path = os.path.join(self.save_dir, load_filename)
            net = getattr(self, model)
            # TODO: Restore checkpoints on MPS
            net.load_state_dict(torch.load(load_path, map_location=self.device))

        if restore_training:
            optimizers = ['optimizer_G', 'optimizer_D']
            for optimizer in optimizers:
                if best_model:
                    load_filename = f"{optimizer}_best.pt"
                else:
                    load_filename = f"{optimizer}_last.pt"
                load_path = os.path.join(self.save_dir, load_filename)
                net = getattr(self, optimizer)
                net.load_state_dict(torch.load(load_path, map_location=self.device))