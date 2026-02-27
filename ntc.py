import math
import time
from pathlib import Path

import cv2

import image_utils as iu
import numpy as np
import torch
import torch.nn as nn
OUTPUT_DIM = 1
EPOCHS = 1

class NTC(nn.Module):
    def __init__(self, feature_dim=16, base_res=256, hidden_dim=32, device='cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        self.batch_size = max(4096//2, (2**14)*(self.device == "cuda"))
        print(f"  - Levels: {base_res}x{base_res} (Features: {feature_dim})")
        self.grid = nn.Parameter(torch.randn(1, feature_dim, base_res, base_res) * 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        grid_params = self.grid.numel()
        mlp_params = sum(p.numel() for p in self.mlp.parameters())
        total_params = grid_params+mlp_params
        print(f"Grid parameters: {grid_params:,}\nMLP parameters: {mlp_params:,}\nTotal parameters: {total_params:,}")
        # print(f"Model Parameters: {total_params:,}, {grid_params+mlp_params:,}")
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
        # print(img.shape)
        # print(f"Using {self.device}")
        indexes = self._create_coords(img.shape[0], img.shape[1])
        X = indexes.to(self.device)
        y = colors.to(self.device)
        num_pixels = X.shape[0]
        # print(num_pixels)

        steps = math.ceil(num_pixels / self.batch_size)
        # print(f"steps: {steps}")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
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
                end_idx = min(start_idx + self.batch_size, num_pixels-1)
                indices = perm[start_idx:end_idx]
                batch_X = X[indices]
                batch_y = y[indices]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device):
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y)
                epoch_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            print(f"Epoch {epoch:3d}, Loss: {epoch_loss:.5f}, Time: {time.time() - ep_start:.3f}")
        print(f"Training done in {time.time() - start:.3f} seconds")


    def generate(self, h,w):
        num_pixels = h*w
        coords = self._create_coords(h,w).to(self.device)
        steps = math.ceil(num_pixels / self.batch_size)
        colors = torch.zeros((num_pixels, OUTPUT_DIM))
        for i in range(steps):
            start_idx = i * self.batch_size
            end_idx = min(start_idx + self.batch_size, num_pixels - 1)
            colors[start_idx:end_idx] = self(coords[start_idx:end_idx])
        colors = colors.cpu().detach().numpy().reshape((h,w, OUTPUT_DIM))
        return colors





img_path = Path("converted/r1_u16rgb.png")
tensor = iu.tensor_float_load(img_path)
h,w = tensor.shape[:2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = NTC(feature_dim=16, base_res=256, hidden_dim=32,device=dev)
model.fit(tensor)
gen = model.generate(h,w)
iu.save_image(img_path.stem, gen.astype(dtype=np.uint8), colorspace="Y", generated=True)