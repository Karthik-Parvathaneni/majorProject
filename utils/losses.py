# utils/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# ---------------------------
# Perceptual Loss
# ---------------------------
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(pretrained=True).features

        self.blocks = nn.ModuleList([
            vgg[:4],    # relu1_2
            vgg[4:9],   # relu2_2
            vgg[9:16],  # relu3_3
        ])

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        loss = 0
        for block in self.blocks:
            x = block(x)
            y = block(y)
            loss += F.l1_loss(x, y)
        return loss


# ---------------------------
# Gradient (Edge) Loss
# ---------------------------
class GradientLoss(nn.Module):
    def __init__(self):
        super().__init__()
        sobel_x = torch.tensor([[1, 0, -1],
                                [2, 0, -2],
                                [1, 0, -1]]).float().unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[1, 2, 1],
                                [0, 0, 0],
                                [-1, -2, -1]]).float().unsqueeze(0).unsqueeze(0)

        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    def rgb_to_gray(self, x):
        # x: [B, 3, H, W]
        r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        return gray


    def forward(self, pred, gt):
        pred_gray = self.rgb_to_gray(pred)
        gt_gray   = self.rgb_to_gray(gt)

        grad_pred_x = F.conv2d(pred_gray, self.sobel_x, padding=1)
        grad_pred_y = F.conv2d(pred_gray, self.sobel_y, padding=1)

        grad_gt_x = F.conv2d(gt_gray, self.sobel_x, padding=1)
        grad_gt_y = F.conv2d(gt_gray, self.sobel_y, padding=1)

        loss = torch.mean(torch.abs(grad_pred_x - grad_gt_x)) + \
            torch.mean(torch.abs(grad_pred_y - grad_gt_y))
        return loss

