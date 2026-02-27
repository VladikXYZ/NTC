import math
import time
from pathlib import Path

import image_utils as iu
import numpy as np
import torch
import torch.nn as nn


"""
Using GPU: NVIDIA GeForce RTX 4070 Laptop GPU
Converting RGB -> YCoCg...
Initializing Feature Pyramid with 7 levels:
  - Level 0: 16x16 (Features: 4)
  - Level 1: 32x32 (Features: 4)
  - Level 2: 64x64 (Features: 4)
  - Level 3: 128x128 (Features: 4)
  - Level 4: 256x256 (Features: 4)
  - Level 5: 512x512 (Features: 4)
  - Level 6: 1024x1024 (Features: 4)
Model Parameters: 5,592,851
Training new model...
torch.compile not available, skipping.
Epoch 001/25 | Loss: 0.000462 | PSNR: 33.35dB
Epoch 005/25 | Loss: 0.000160 | PSNR: 37.95dB
Epoch 010/25 | Loss: 0.000136 | PSNR: 38.66dB
Epoch 015/25 | Loss: 0.000126 | PSNR: 39.01dB
Epoch 020/25 | Loss: 0.000120 | PSNR: 39.22dB
Epoch 025/25 | Loss: 0.000116 | PSNR: 39.37dB
Training finished in 120.28s
Model saved to latent_model.pth
"""


class NTC(nn.Module):
    def __init__(self, feature_dim=28, base_res=256, hidden_dim=32, device='cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.batch_size = max(4096//2, (2**17)*(self.device == "cuda"))
        print(f"Grid: {base_res}x{base_res} (Features: {feature_dim})\nOutput dim: {OUTPUT_DIM}")
        self.grid = nn.Parameter(torch.randn(1, feature_dim, base_res, base_res) * 0.01)
        self.mlp = nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, OUTPUT_DIM), nn.Sigmoid())
        grid_params = self.grid.numel()
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        total_params = grid_params+mlp_params
        print(f"Grid parameters: {grid_params:,}\nMLP parameters: {mlp_params:,}\nTotal parameters: {total_params:,}")
        self.to(device)

    def forward(self, coords):
        coords = coords.view(1, 1, -1, 2)
        features =  torch.nn.functional.grid_sample( self.grid, coords, mode='bilinear',
                padding_mode='border', align_corners=True )
        features = features.squeeze().permute(1, 0).contiguous()
        colors = self.mlp(features)
        return colors

    def _create_coords(self, h,w):
        xs, ys = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
        indexes = np.stack((ys.flatten(), xs.flatten()), axis=-1)
        return torch.as_tensor(indexes, dtype=torch.float32)

    def fit(self, img):
        if img.dtype != torch.float32: raise TypeError(f"img must be np.float32 not {img.dtype}")
        colors = img.reshape(-1, OUTPUT_DIM)
        indexes = self._create_coords(img.shape[0], img.shape[1])
        X = indexes.to(self.device)
        y = colors.to(self.device)
        num_pixels = X.shape[0]

        steps = math.ceil(num_pixels / self.batch_size)
        optimizer = torch.optim.Adam(self.parameters(), lr=0.005)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(self.device)
        model.train()
        print(f"Training on {self.device} started ...")
        print(f"Epochs: {EPOCHS}\nData size: {num_pixels:,}\nBatch size: {self.batch_size:,}\nSteps: {steps:,}")
        start = time.time()
        for epoch in range(EPOCHS):
            ep_start = time.time()
            epoch_loss = 0
            perm = torch.randperm(num_pixels, device=self.device)
            for i in range(steps):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_pixels)
                indices = perm[start_idx:end_idx]
                batch_X = X[indices]
                batch_y = y[indices]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device):
                    preds = self(batch_X)
                    loss = criterion(batch_y, preds)


                epoch_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()


            psnr = 20 * np.log10(1 / np.sqrt(epoch_loss/steps))
            print(f"Epoch {epoch:3d}, Loss: {epoch_loss:.5f}, PSNR: {psnr:.2f}, Time: {time.time() - ep_start:.3f}")
        print(f"Training done in {time.time() - start:.3f} seconds")


    def generate(self, h,w):
        num_pixels = h*w
        model.eval()
        coords = self._create_coords(h,w).to(self.device)
        steps = math.ceil(num_pixels / self.batch_size)
        colors = torch.zeros((num_pixels, OUTPUT_DIM))
        with torch.no_grad():
            for i in range(steps):
                start_idx = i * self.batch_size
                end_idx = min(start_idx + self.batch_size, num_pixels)
                colors[start_idx:end_idx] = self(coords[start_idx:end_idx])
        colors = colors.reshape((h,w, OUTPUT_DIM)).cpu().detach().numpy()
        return colors





img_path = Path(f"converted/r4_u8{COLOR_SPACE}.png")
tensor = iu.tensor_float_load(img_path)
h,w = tensor.shape[:2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = NTC(feature_dim=16, base_res=256, hidden_dim=16,device=dev)

model.fit(tensor)
gen = model.generate(1024,1024)
iu.save_image(img_path.stem, gen, colorspace=COLOR_SPACE, generated=True)