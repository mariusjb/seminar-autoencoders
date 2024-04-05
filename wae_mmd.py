import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import Decoder
from discriminator import LatentDiscriminator
from encoder import Encoder
import geomloss


class WAE_MMD(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim, lmbda, kernel, device='mps'):
        super(WAE_MMD, self).__init__()
        torch.autograd.set_detect_anomaly(True)

        self.device = device

        self.z_dim = z_dim

        self.encoder = Encoder(in_channels, out_channels, z_dim).to(self.device)
        self.decoder = Decoder(in_channels, out_channels, z_dim).to(self.device)
        self.latent_discriminator = LatentDiscriminator(z_dim).to(self.device)
        self.lambda_value = lmbda
        self.kernel = kernel

    def prior_z(self):
        z_mean = torch.zeros(self.z_dim, device=self.device)
        z_var = torch.ones(self.z_dim, device=self.device)
        return  torch.distributions.Independent( torch.distributions.Normal(z_mean, z_var), 1)

    def compute_kernel(self, x, y):
        return self.kernel(x, y)

    def compute_mmd_loss(self, z_prior, z_encoded):
        k_prior_prior = self.compute_kernel(z_prior.unsqueeze(1), z_prior.unsqueeze(0))
        k_prior_encoded = self.compute_kernel(z_prior.unsqueeze(1), z_encoded.unsqueeze(0))
        k_encoded_encoded = self.compute_kernel(z_encoded.unsqueeze(1), z_encoded.unsqueeze(0))

        mmd_loss = torch.mean(k_prior_prior) + torch.mean(k_encoded_encoded) - 2 * torch.mean(k_prior_encoded)
        return mmd_loss
    
    def compute_wasserstein_loss(self, z_prior, z_encoded):
        # Compute the Wasserstein distance using Sinkhorn algorithm
        loss = geomloss.SamplesLoss("sinkhorn", p=2, blur=0.05)
        w_loss = loss(z_prior, z_encoded)
        return w_loss
    
    def compute_kl_divergence(self, z_mean, z_logvar):
        # Compute KL divergence between Gaussian distributions
        kl_divergence = -0.5 * torch.sum(1 + z_logvar - z_mean.pow(2) - z_logvar.exp(), dim=1)
        return kl_divergence.mean()

    def forward(self, x):
        z_mean, z_logvar = self.encoder(x.to(self.device))
        z = self.reparameterize(z_mean, z_logvar)
        x_hat = self.decoder(z.to(self.device))
        return x_hat, z_mean, z_logvar

    def reparameterize(self, mean, log_var):
        std = torch.exp(0.5 * log_var)
        epsilon = torch.randn_like(std, device=self.device)
        z = mean + std * epsilon
        return z

    def train_step(self, x, encoder_optimizer, decoder_optimizer, latent_discriminator_optimizer):
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

        z_mean, z_logvar = self.encoder(x.to(self.device))
        z_encoded = self.reparameterize(z_mean, z_logvar)
        z_prior = self.prior_z().sample((x.size(0),)).to(self.device)
        x_hat = self.decoder(z_encoded)

        # Compute loss
        reconstruction_loss = F.mse_loss(x_hat, x)

        # MMD Loss
        mmd_loss = self.compute_mmd_loss(z_prior, z_encoded)
        total_loss = reconstruction_loss + self.lambda_value * mmd_loss

        # Wasserstein Loss
        # wasserstein_loss = self.compute_wasserstein_loss(z_prior, z_encoded)
        # total_loss = reconstruction_loss + self.lambda_value * wasserstein_loss

        # KL Divergence
        # kl_loss = self.compute_kl_divergence(z_mean, z_logvar)
        # total_loss = reconstruction_loss + self.lambda_value * kl_loss

        # Backpropagation for encoder and decoder
        total_loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()

        # Update Latent Discriminator
        latent_discriminator_loss = self.update_latent_discriminator(z_prior.detach(), z_encoded.detach(), latent_discriminator_optimizer)

        return reconstruction_loss.item(), mmd_loss.item(), latent_discriminator_loss.item()

    def update_latent_discriminator(self, z_prior, z_encoded, latent_discriminator_optimizer):
        latent_discriminator_optimizer.zero_grad()
        z_prior_pred = self.latent_discriminator(z_prior)
        z_encoded_pred = self.latent_discriminator(z_encoded)

        latent_discriminator_loss = F.binary_cross_entropy_with_logits(z_prior_pred, torch.ones_like(z_prior_pred).to(self.device)) + \
                                    F.binary_cross_entropy_with_logits(z_encoded_pred, torch.zeros_like(z_encoded_pred).to(self.device))
        latent_discriminator_loss.backward()
        latent_discriminator_optimizer.step()

        return latent_discriminator_loss
