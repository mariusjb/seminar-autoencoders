from dataclasses import dataclass

@dataclass
class Config:

    # Data
    dataset_name: str = 'celebA'
    img_height: int = 28
    img_width: int = 28
    num_channels: int = 1

    z_dim: int = 8

    
