

# Wasserstein Auto-Encoder

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class WAE(torch.nn.Module):
    def __init__(self, input_size, latent_dim):
        super(WAE, self).__init__()
        self.encoder = torch.nn.Sequential(
            torch.nn.Flatten(),
            torch.nn.Linear(input_size, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU()
        )
        self.mean = torch.nn.Linear(256, latent_dim)
        self.log_var = torch.nn.Linear(256, latent_dim)

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(latent_dim, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, input_size),
            torch.nn.Sigmoid()
        )

    def encode(self, x):
        """
        Encode the input x to obtain mean and log variance of the latent variable z.
        """
        x = self.encoder(x)
        mean = self.mean(x)
        log_var = self.log_var(x)
        return mean, log_var
    
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

def load_data():
    """
    Load the MNIST dataset and return data loaders for training and testing.
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    return train_loader, test_loader

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

def train(wae, train_loader, optimizer, num_epochs=10, device='cpu'):
    wae.to(device)
    wae.train()
    
    for epoch in range(num_epochs):
        total_loss = 0.0
        for batch_idx, (data, _) in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            x_hat, mu, log_var = wae(data)
            loss = wae_loss(x_hat, data, mu, log_var)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}')
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')


def test_wae(wae, test_loader, device='mps'):
    wae.to(device)
    wae.eval()
    
    total_loss = 0.0
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device)
            reconstructed_data, mu, log_var = wae(data)
            loss = wae_loss(reconstructed_data, data, mu, log_var)
            total_loss += loss.item()
    
    avg_loss = total_loss / len(test_loader.dataset)
    print(f'Test Average Loss: {avg_loss:.4f}')


# # Load the MNIST dataset
# train_loader, test_loader = load_data()

# # Initialize WAE model
# input_size = 28 * 28
# latent_dim = 10
# wae = WAE(input_size, latent_dim)

# # Train the WAE model
# optimizer = torch.optim.Adam(wae.parameters(), lr=1e-3)
# num_epochs = 10
# train(wae, train_loader, optimizer, num_epochs)

# # Test the WAE model
# test_wae(wae, test_loader)


# # Save the trained model
# torch.save(wae.state_dict(), 'wae.pth')
