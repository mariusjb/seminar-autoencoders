from dataclasses import dataclass
import logging
import sys
import time
import numpy as np
import torch
from tqdm import tqdm
import torch.nn as nn

from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # Layer 1 - 28x28xin -> 14x14x16
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 2 - 14x14x16 -> 7x7x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3 - 7x7x32 -> 4x4xout
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.mu = nn.Linear(out_channels * 4 * 4, z_dim)
        self.logvar = nn.Linear(out_channels * 4 * 4, z_dim)


class WAE(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, stddev=0.02):
        super(WAE, self).__init__()

        self.encoder = nn.Sequential(
            # Layer 1 - 28x28xin -> 14x14x16
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 2 - 14x14x16 -> 7x7x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3 - 7x7x32 -> 4x4xout
            nn.Conv2d(32, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.mu = nn.Linear(out_channels * 4 * 4, z_dim)
        self.logvar = nn.Linear(out_channels * 4 * 4, z_dim)

        self.decoder = nn.Sequential(
            # Layer 1 - 4x4xout -> 7x7x32
            nn.ConvTranspose2d(out_channels, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2 - 7x7x32 -> 14x14x16
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 3 - 14x14x16 -> 28x28xin
            nn.ConvTranspose2d(16, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()  # Ensure output values are in the range [0, 1] for image data
        )


    def prior_z(latent_dim):
        z_mean = torch.zeros(latent_dim)
        z_var = torch.ones(latent_dim)
        return torch.distributions.Independent(torch.distributions.Normal(z_mean, torch.diag(z_var)))

        prior_dist = prior_z(z_dim);

        z_prime = prior_dist.sample(batch_size);


    def encode(self, x):
        """
        Encode the input x to obtain mu and log variance of the latent variable z.
        """
        x = self.encoder(x)
        mu = self.mu(x)
        log_var = self.logvar(x)
        return mu, log_var
    
    def reparameterize(self, mean, log_var):
        """
        Reparameterization trick for sampling from Gaussian distribution.
        """
        std = torch.exp(0.5 * log_var)  # Standard deviation (sigma)
        epsilon = torch.randn_like(std)  # Sample from standard Gaussian distribution
        z = mean + std * epsilon  # Reparameterization trick
        return z
    
    def decode(self, z):
        """
        Decode the latent variable z to reconstruct the input.
        """
        return self.decoder(z)
    
    def forward(self, x):
        """
        Forward pass of the WAE model.
        """
        mean, log_var = self.encode(x)
        z = self.reparameterize(mean, log_var)
        x_hat = self.decode(z)
        return x_hat, mean, log_var

class ModelRunner:
    def __init__(self, model: torch.nn.Module, configs, optimizer, loss_fn, data):
        self.model = model
        self.configs = configs
        self.optimizer = optimizer
        self.loss_fn = loss_fn

        self.train_loader = None
        self.test_loader = None


    def set_data(self, dataset_name: str, batch_size: int, shuffle: bool = True):
        logger.info(f"Loading dataset: {dataset_name}")
        dataset = getattr(datasets, dataset_name)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])

        logger.info(f"Setting up train data loader...")
        train_dataset = dataset(root='./data', train=True, download=True, transform=transform)
        self.train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)

        logger.info(f"Setting up test data loader...")
        test_dataset = dataset(root='./data', train=False, download=True, transform=transform)
        self.test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    def train(self, num_epochs: int = 10, device: str = 'cpu', loss_window_size: int = 20):
        logger.info(f"Training the model for {num_epochs} epochs...")
        self.model.to(device)
        self.model.train()
        
        num_batches = len(self.train_loader)
        total = num_epochs * num_batches
        
        with tqdm(total=total, desc="Training") as pbar:
            iteration = 0
            moving_losses = []
            moving_loss = 0.0
            for epoch in range(num_epochs):
                for batch_idx, (X, _) in enumerate(self.train_loader):
                    X = X.to(device)
                    self.optimizer.zero_grad()
                    X_hat, mu, log_var = self.model(X)
                    loss = self.loss_fn(X_hat, X, mu, log_var)
                    loss.backward()
                    self.optimizer.step()

                    # # Update moving average for loss
                    moving_losses.append(loss.item())
                    if len(moving_losses) > loss_window_size:
                        moving_losses.pop(0)

                    if iteration % loss_window_size == 0:
                        moving_loss = np.mean(moving_losses)
                    # if pbar.n >= loss_window_size:
                    #     moving_loss = moving_losses.mean().item()

                    pbar.update(1)
                    pbar.set_postfix({'Epoch': f'{epoch+1}/{num_epochs}', 
                                    'Batch': f'{batch_idx+1}/{num_batches}',
                                    'Moving Loss': f'{moving_loss:.4f}'
                                    })
                    
                    iteration += 1

        # logger.info(f'Final Moving Average Loss: {moving_loss:.4f}')
        logger.info(f"Finishing training...")


def wae_loss(x_hat, x, mu, log_var, lmda=1.0):
    """
    Computes the Wasserstein Autoencoder (WAE) loss.

    Args:
        x_hat: Reconstructed input data.
        x: Original input data.
        mu: Mean of the latent distribution.
        log_var: Logarithm of the variance of the latent distribution.
        lmda: Weight for the KL divergence term.

    Returns:
        loss: WAE loss value.
    """
    # Reshape the reconstructed_x to match the original input shape
    x_hat = x_hat.view(x.size())
    
    # Reconstruction loss (Mean Squared Error)
    recon_loss = torch.nn.functional.mse_loss(x_hat, x, reduction='mean')

    # KL divergence between the latent distribution and standard Gaussian
    kl_divergence = -0.5 * torch.mean(1 + log_var - mu.pow(2) - log_var.exp())

    # Total WAE loss
    loss = recon_loss + lmda * kl_divergence
    return loss


mdl = WAE(28*28, 8)
optimizer = torch.optim.Adam(mdl.parameters(), lr=1e-3)
loss_fn = wae_loss

mdl_runner = ModelRunner(mdl, None, optimizer, loss_fn, None)
mdl_runner.set_data('MNIST', batch_size=128, shuffle=True)
mdl_runner.train(num_epochs=2, device='mps')





    



    # def train(self, num_epochs=10, device='cpu'):
    #     n_batches = 

    #     self.model.to(device)
    #     self.model.train()
        
    #     for epoch in range(num_epochs):
    #         total_loss = 0.0
    #         for batch_idx, (data, _) in enumerate(self.data_loader):
    #             data = data.to(device)
    #             self.optimizer.zero_grad()
    #             x_hat, mu, log_var = self.model(data)
    #             loss = self.loss_fn(x_hat, data, mu, log_var)
    #             loss.backward()
    #             self.optimizer.step()
    #             total_loss += loss.item()
                
    #             if batch_idx % 100 == 0:
    #                 print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(self.data_loader)}], Loss: {loss.item():.4f}')
            
    #         print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(self.data_loader):.4f}')