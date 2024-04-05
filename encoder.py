from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, z_dim):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.z_dim = z_dim
    
        self.encoder = nn.Sequential(
            # Layer 1 - 28x28xin -> 14x14x16
            nn.Conv2d(self.in_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),

            # Layer 2 - 14x14x16 -> 7x7x32
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            # Layer 3 - 7x7x32 -> 4x4xout
            nn.Conv2d(32, self.out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.out_channels),
            nn.ReLU()
        )

        # Calculate the size of the output of the last convolutional layer
        # This will be used to define the fully connected layers
        self.flatten_size = self.out_channels * 4 * 4  # 28x28x1 reduced to 4x4xout

        self.fc_mu = nn.Linear(self.flatten_size, z_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, z_dim)

    def forward(self, x):
        x = self.encoder(x)
        x = x.view(-1, self.flatten_size)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        return mu, logvar


# from torch import nn

# class Encoder(nn.Module):
#     def __init__(self, dshape, z_dim, n_layers=3):
#         super(Encoder, self).__init__()
#         self.dshape = dshape
#         self.z_dim = z_dim

        

#         for i in range(n_layers):
#             if i == 0:
#                 self.add_module(f'conv{i}', nn.Conv2d(dshape[0], 16, kernel_size=3, stride=2, padding=1))
#             else:
#                 self.add_module(f'conv{i}', nn.Conv2d(16*(2**(i-1)), 16*(2**i), kernel_size=3, stride=2, padding=1))
#             self.add_module(f'bn{i}', nn.BatchNorm2d(16*(2**i)))
#             self.add_module(f'relu{i}', nn.ReLU())

#         # Calculate the size of the output of the last convolutional layer
#         # This will be used to define the fully connected layers
#         self.flatten_size = 16*(2**(n_layers-1)) * (dshape[1]//(2**n_layers)) * (dshape[2]//(2**n_layers))

#         self.fc_mu = nn.Linear(self.flatten_size, z_dim)
#         self.fc_logvar = nn.Linear(self.flatten_size, z_dim)

#     def forward(self, x):
#         x = self(x)
#         x = x.view(-1, self.flatten_size)

#         mu = self.fc_mu(x)
#         logvar = self.fc_logvar(x)

#         return mu, logvar
