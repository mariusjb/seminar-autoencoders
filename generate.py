import torch
from wae import WAE
from matplotlib import pyplot as plt

def generate_images(wae, num_images=10, latent_dim=10):
    wae.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        latent_vectors = torch.randn(num_images, latent_dim)
        generated_images = wae.decode(latent_vectors)
    return generated_images

if __name__ == "__main__":
    input_size = 28 * 28
    latent_dim = 10
    wae = WAE(input_size, latent_dim)
    wae.load_state_dict(torch.load('wae.pth'))
    generated_images = generate_images(wae)
    fig, axes = plt.subplots(1, len(generated_images), figsize=(15, 3))
    for i, image in enumerate(generated_images):
        axes[i].imshow(image.view(28, 28).cpu(), cmap='gray')
        axes[i].axis('off')
    plt.show()