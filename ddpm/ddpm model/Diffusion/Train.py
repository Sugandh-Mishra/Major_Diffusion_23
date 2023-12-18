import os
from typing import Dict

import torch
import shutil
import pickle
import numpy as np

from PIL import Image
import torch.optim as optim
from torchvision import transforms
from torchvision.transforms import Resize
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.utils import save_image
import torch.nn.functional as F

from Diffusion import GaussianDiffusionSampler, GaussianDiffusionTrainer
from Diffusion.Model import UNet
from Scheduler import GradualWarmupScheduler
from torchvision.models import inception_v3
from Diffusion.cifarimg import save_random_cifar_images

def save_images(images, save_folder, prefix='sample_images_'):
    os.makedirs(save_folder, exist_ok=True)
    for i, image in enumerate(images):
        save_path = os.path.join(save_folder, f'{prefix}{i}.png')
        save_image(image, save_path)

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
                aux_output = None 
            elif isinstance(output, tuple) and len(output) == 2:
                logits, aux_output = output
            else:
                raise ValueError("Unexpected output structure from the inception_model")
        
        all_logits.append(logits)

    all_logits = torch.cat(all_logits, dim=0)
    probs = torch.nn.functional.softmax(logits, dim=1)
    
    # Use torchmetrics to calculate Inception Score
    is_score = torch.mean(torch.sum(probs * torch.log2(1.0 / probs), dim=1))

    return is_score.item()

def train(modelConfig: Dict):
    device = torch.device(modelConfig["device"])
    # dataset
    dataset = CIFAR10(
        root='./CIFAR10', train=True, download=True,
        transform=transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]))
    dataloader = DataLoader(
        dataset, batch_size=modelConfig["batch_size"], shuffle=True, num_workers=4, drop_last=True, pin_memory=True)

    # model setup
    net_model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=modelConfig["dropout"]).to(device)
    if modelConfig["training_load_weight"] is not None:
        net_model.load_state_dict(torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["training_load_weight"]), map_location=device))
    optimizer = torch.optim.AdamW(
        net_model.parameters(), lr=modelConfig["lr"], weight_decay=1e-4)
    cosineScheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer, T_max=modelConfig["epoch"], eta_min=0, last_epoch=-1)
    warmUpScheduler = GradualWarmupScheduler(
        optimizer=optimizer, multiplier=modelConfig["multiplier"], warm_epoch=modelConfig["epoch"] // 10, after_scheduler=cosineScheduler)
    trainer = GaussianDiffusionTrainer(
        net_model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)

    # start training
    for e in range(modelConfig["epoch"]):
        with tqdm(dataloader, dynamic_ncols=True) as tqdmDataLoader:
            for images, labels in tqdmDataLoader:
                # train
                optimizer.zero_grad()
                x_0 = images.to(device)
                loss = trainer(x_0).sum() / 1000.
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    net_model.parameters(), modelConfig["grad_clip"])
                optimizer.step()
                tqdmDataLoader.set_postfix(ordered_dict={
                    "epoch": e,
                    "loss: ": loss.item(),
                    "img shape: ": x_0.shape,
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"]
                })
        warmUpScheduler.step()
        torch.save(net_model.state_dict(), os.path.join(
            modelConfig["save_weight_dir"], 'ckpt_' + str(e) + "_.pt"))


def eval(modelConfig: Dict):
    # load model and evaluate
    args = {
        "exp": "/home/shavak/Desktop/Finalpro/ddp",
        # Add any other necessary parameters
    }
    config = {
        "data": {
            "dataset": "CIFAR10",  # or "CELEBA"
            "exp": "./Experiments/"
          }
      }
    with torch.no_grad():
        device = torch.device(modelConfig["device"])
        model = UNet(T=modelConfig["T"], ch=modelConfig["channel"], ch_mult=modelConfig["channel_mult"], attn=modelConfig["attn"],
                     num_res_blocks=modelConfig["num_res_blocks"], dropout=0.)
        ckpt = torch.load(os.path.join(
            modelConfig["save_weight_dir"], modelConfig["test_load_weight"]), map_location=device)
        model.load_state_dict(ckpt)
        print("model load weight done.")
        model.eval()
        sampler = GaussianDiffusionSampler(
            model, modelConfig["beta_1"], modelConfig["beta_T"], modelConfig["T"]).to(device)
        
        # Sampled from standard normal distribution
        noisy_images = torch.randn(
            size=[modelConfig["batch_size"], 3, 32, 32], device=device)
        # saveNoisy = torch.clamp(noisyImage * 0.5 + 0.5, 0, 1)
        # save_image(saveNoisy, os.path.join(
        #     modelConfig["sampled_dir"], modelConfig["sampledNoisyImgName"])+, nrow=modelConfig["nrow"])
        
        # Generate sampled images
        sampled_images = []
        for i in range(0, modelConfig["batch_size"], modelConfig["batch_size_per_iteration"]):
            batch_noisy_images = noisy_images[i:i + modelConfig["batch_size_per_iteration"]]
            sampled_images.extend(sampler(batch_noisy_images))
        # sampledImgs = sampler(noisyImage)
        # sampledImgs = sampledImgs * 0.5 + 0.5  # [0 ~ 1]
        print("****************************")
        print(len(sampled_images))
        # print(config["data"]["dataset"])
        # sampledImgs = torch.clamp(sampledImgs, 0, 1)
        

    # Save the resized image
        save_path = os.path.join(modelConfig["sampled_dir"])
        save_images(sampled_images, save_path)
        save_random_cifar_images(modelConfig["batch_size"],cifar10_path='CIFAR10/cifar-10-batches-py',save_folder='original')
        # Calculate FID
        # fid_path = get_fid_stats_path(args, config=config, download=True)
        # fid_value = get_fid(fid_path, os.path.join(modelConfig["sampled_dir"]))
        
        # print(f"FID Score: {fid_value}")
