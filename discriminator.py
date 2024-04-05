from torch import nn

class LatentDiscriminator(nn.Module):
    def __init__(self, z_dim):
        super(LatentDiscriminator, self).__init__()
        self.z_dim = z_dim

        self.discriminator = nn.Sequential(
            nn.Linear(self.z_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, z):
        return self.discriminator(z)