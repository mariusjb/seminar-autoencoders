from torch import nn

class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim):
        super(Decoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_dim = z_dim

        self.fc = nn.Linear(self.z_dim, self.out_channels * 4 * 4)

        self.decoder = nn.Sequential(
            # Layer 1 - 4x4xout -> 7x7x32
            nn.ConvTranspose2d(self.out_channels, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 2 - 7x7x32 -> 14x14x16
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 3 - 14x14x16 -> 28x28xin
            nn.ConvTranspose2d(16, self.in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(self.in_channels),
            nn.Sigmoid()  # Ensure output values are in the range [0, 1] for image data
        )

    def forward(self, z):
        x = self.fc(z)
        x = x.view(-1, self.out_channels, 4, 4)  # Reshape the tensor for convolutional layers
        x = self.decoder(x)
        return x
