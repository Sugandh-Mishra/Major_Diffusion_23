import os
import random
from PIL import Image
from torchvision import datasets
from pytorch_fid import fid_score

# Set the path to your CIFAR-10 dataset
cifar10_path = '/content/drive/MyDrive/ddp/CIFAR10/cifar-10-batches-py'

# Set the path to your generated images

# Load CIFAR-10 dataset
save_folder = 'original'
cifar10_dataset = datasets.CIFAR10(root=cifar10_path, download=True)
# print(len(cifar10_dataset))
os.makedirs(save_folder, exist_ok=True)

num_images_to_save = 10

random_indices = random.sample(range(len(cifar10_dataset)), num_images_to_save)

# Save individual CIFAR-10 images
for i, index in enumerate(random_indices):
    cifar10_image = cifar10_dataset.data[index]
    image = Image.fromarray(cifar10_image)
    save_path = os.path.join(save_folder, f'cifar10_image_{i}.png')
    image.save(save_path)
# use libraries where we just need to supply path for original and SampledImgs folder it automatically calculates its fid
# Load and check generated images
# generated_images = load_images(generated_images_path)
# check_images(generated_images)

# Print the dimensions of CIFAR-10 images
# cifar10_image = cifar10_dataset.data[0]
# cifar10_image_size = Image.fromarray(cifar10_image).size
# print(f"Dimensions of CIFAR-10 images: {cifar10_image_size}")

# # Calculate FID score
# fid_value = fid_score.calculate_fid_given_paths(
#     [cifar10_dataset.data],  # Real data
#     generated_images,           # Generated data
#     batch_size=10,
#     dims=2048
# )

# print(f"FID Score: {fid_value}")
