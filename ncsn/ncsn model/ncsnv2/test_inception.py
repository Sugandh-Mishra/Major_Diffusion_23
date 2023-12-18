import os
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from PIL import Image

def custom_read_image(file_path):
    # Use PIL to read the image
    return Image.open(file_path).convert("RGB")

def calculate_inception_score(images, inception_model, batch_size=64, resize=False):
    inception_model.eval()
    all_logits = []

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            if resize:
                batch = F.interpolate(batch, size=(299, 299), mode='bilinear', align_corners=False)
            output = inception_model(batch)
            if isinstance(output, torch.Tensor):
                logits = output
            elif isinstance(output, tuple) and len(output) == 2:
                logits, _ = output
            else:
                raise ValueError("Unexpected output structure from the inception_model")
        
            all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.nn.functional.softmax(all_logits, dim=1)
    
    # Use torchmetrics to calculate Inception Score
    is_score = torch.mean(torch.sum(probs * torch.log2(1.0 / probs), dim=1))

    return is_score.item()

def main():
    # Replace with the path to your image folder
    image_folder_path = 'img_test/ckpt_300000'

    # Set up data transformation
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.Lambda(custom_read_image),  # Use custom_read_image to read images
        transforms.ToTensor(),
    ])

    # Create a dataset from the image folder
    dataset = ImageFolder(root=image_folder_path, transform=transform, loader=custom_read_image)

    # Create a DataLoader to handle batching
    dataloader = DataLoader(dataset, batch_size=64, shuffle=False)

    # Load or instantiate your InceptionV3 model
    inception_model = ...

    all_images = []

    for images, _ in dataloader:
        all_images.append(images)

    all_images = torch.cat(all_images, dim=0)

    # Calculate Inception Score
    is_score = calculate_inception_score(all_images, inception_model, batch_size=64, resize=False)
    print(f"Inception Score: {is_score}")

if __name__ == "__main__":
    main()
