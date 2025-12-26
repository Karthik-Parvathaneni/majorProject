import time
import torch
import argparse
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.train_data_functions import TrainData
from utils.val_data_functions import ValData
from utils.metrics import calculate_psnr, calculate_ssim
import os
import numpy as np
import random
import torchvision.utils as tvu
import cv2
from model.cmformer import CMFormer
from utils.losses import VGGPerceptualLoss, GradientLoss


# =========================
# Argument parser
# =========================
parser = argparse.ArgumentParser(description='Hyper-parameters for network')
parser.add_argument('-learning_rate', default=2e-4, type=float)
parser.add_argument('-crop_size', default=[128, 128], nargs='+', type=int)
parser.add_argument('-train_batch_size', default=8, type=int)
parser.add_argument('-val_batch_size', default=1, type=int)
parser.add_argument('-num_steps', default=90000, type=int)
parser.add_argument('-save_step', default=2000, type=int)
parser.add_argument('-exp_name', type=str)
parser.add_argument('-seed', default=19, type=int)
parser.add_argument('-train_data_dir', type=str)
parser.add_argument('-val_data_dir', type=str)
parser.add_argument('-checkpoint', type=str, default=None)

args = parser.parse_args()


# =========================
# Setup
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs(args.exp_name, exist_ok=True)

def save_image(img, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tvu.save_image(img, path)


# =========================
# Reproducibility
# =========================
torch.manual_seed(args.seed)
np.random.seed(args.seed)
random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)


# =========================
# Model & optimizer
# =========================
net = CMFormer().to(device)
print("Number of parameters: %.2fM" %
      (sum(p.numel() for p in net.parameters()) / 1e6))

optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)


# =========================
# Loss functions
# =========================
pixel_loss_fn = nn.L1Loss()
perceptual_loss_fn = VGGPerceptualLoss().to(device)
gradient_loss_fn = GradientLoss().to(device)

# Freeze VGG perceptual network
for p in perceptual_loss_fn.parameters():
    p.requires_grad = False
perceptual_loss_fn.eval()


# =========================
# Resume from checkpoint
# =========================
total_steps = 0

if args.checkpoint is not None:
    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location=device)
    net.load_state_dict(checkpoint['model'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    total_steps = checkpoint['step']

    print(f" Resumed training from step {total_steps}")


# =========================
# Data loaders
# =========================
train_loader = DataLoader(
    TrainData(args.crop_size, args.train_data_dir, 'train.txt',
              random_flip=True, random_rotate=True),
    batch_size=args.train_batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    ValData(args.val_data_dir, 'test.txt'),
    batch_size=args.val_batch_size,
    shuffle=False,
    num_workers=2
)


def denormalize(x):
    return (x + 1) / 2


# =========================
# Training loop
# =========================
net.train()

while total_steps < args.num_steps:
    for input_image, gt, _ in train_loader:

        input_image = input_image.to(device)
        gt = gt.to(device)

        optimizer.zero_grad()

        pred_image = net(input_image)

        l_pixel = pixel_loss_fn(pred_image, gt)
        l_perc = perceptual_loss_fn(
            denormalize(pred_image),
            denormalize(gt)
        )
        l_grad = gradient_loss_fn(pred_image, gt)

        loss = (
            1.0 * l_pixel +
            0.1 * l_perc +
            0.05 * l_grad
        )

        loss.backward()
        optimizer.step()

        total_steps += 1

        if total_steps % 10 == 0:
            print(f"Steps:{total_steps} | "
                  f"L1:{l_pixel:.4f} | "
                  f"Perc:{l_perc:.4f} | "
                  f"Grad:{l_grad:.4f}")

        # =========================
        # Validation + Metrics + Checkpoint
        # =========================
        if total_steps % args.save_step == 0:
            net.eval()

            eva_out = "./eva/output"
            eva_gt = "./eva/gt"
            os.makedirs(eva_out, exist_ok=True)
            os.makedirs(eva_gt, exist_ok=True)

            start_time = time.time()
            with torch.no_grad():
                for i, (inp, gt, _) in enumerate(val_loader):
                    inp, gt = inp.to(device), gt.to(device)
                    pred = net(inp)
                    save_image(pred, os.path.join(eva_out, f"{i}.png"))
                    save_image(gt, os.path.join(eva_gt, f"{i}.png"))

            test_time = time.time() - start_time
            per_image_time = test_time / (i + 1)

            imgs = sorted(os.listdir(eva_out))
            gts = sorted(os.listdir(eva_gt))

            cumulative_psnr = 0
            cumulative_ssim = 0

            for i in range(len(imgs)):
                res = cv2.imread(os.path.join(eva_out, imgs[i]))
                gt_img = cv2.imread(os.path.join(eva_gt, gts[i]))

                cumulative_psnr += calculate_psnr(res, gt_img, test_y_channel=True)
                cumulative_ssim += calculate_ssim(res, gt_img, test_y_channel=True)

            psnr = cumulative_psnr / len(imgs)
            ssim = cumulative_ssim / len(imgs)

            print(f"\n Evaluation @ step {total_steps}")
            print(f"PSNR: {psnr:.4f} | SSIM: {ssim:.4f}")
            print(f"Test speed: {per_image_time:.4f} sec/image\n")

            # Save evaluation log
            with open("./eva/eva.txt", "a") as f:
                f.write(f"step:{total_steps}, PSNR:{psnr}, SSIM:{ssim}, time:{per_image_time}\n")

            # Save FULL checkpoint
            checkpoint = {
                'step': total_steps,
                'model': net.state_dict(),
                'optimizer': optimizer.state_dict()
            }

            torch.save(checkpoint,
                       os.path.join(args.exp_name, f"{total_steps}_ckpt.pth"))
            torch.save(checkpoint,
                       os.path.join(args.exp_name, "latest.pth"))

            print(f" Checkpoint saved at step {total_steps}")

            net.train()
            torch.cuda.empty_cache()

        if total_steps >= args.num_steps:
            print(" Training finished")
            exit()
