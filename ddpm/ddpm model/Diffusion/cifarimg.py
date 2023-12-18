import os
import random
from torchvision import datasets
from PIL import Image

def save_random_cifar_images(num_images_to_save, cifar10_path='path/to/ddp/CIFAR10/cifar-10-batches-py', save_folder='path/to/ddp/original'):
    # Load CIFAR-10 dataset
    cifar10_dataset = datasets.CIFAR10(root=cifar10_path, download=True)

    # Create the save folder if it doesn't exist
    os.makedirs(save_folder, exist_ok=True)

    # Randomly choose indices for images
    random_indices = random.sample(range(len(cifar10_dataset)), num_images_to_save)

    # Save individual CIFAR-10 images
    for i, index in enumerate(random_indices):
        cifar10_image = cifar10_dataset.data[index]
        image = Image.fromarray(cifar10_image)
        save_path = os.path.join(save_folder, f'cifar10_image_{i}.png')
        image.save(save_path)

    print(f"{num_images_to_save} randomly selected CIFAR-10 images saved in {save_folder}")