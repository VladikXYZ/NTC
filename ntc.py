import math
from pathlib import Path

import cv2
from triton.language import dtype

import image_utils as iu
import numpy as np
import torch
import torch.nn as nn
# COLOR_SPACE = "Y"
OUTPUT_DIM = 1


class NTC(nn.Module):
    def __init__(self, feature_dim=16, base_res=256, hidden_dim=32, device='cpu'):
        super().__init__()
        self.feature_dim = feature_dim
        self.device = device
        print(f"  - Levels: {base_res}x{base_res} (Features: {feature_dim})")
        self.grid = nn.Parameter(torch.randn(1, feature_dim, base_res, base_res) * 0.01)
        self.mlp = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
        total_params = sum(p.numel() for p in self.parameters())
        print(f"Model Parameters: {total_params:,}")
        self.to(device)

    def forward(self, coords):
        features =  torch.nn.functional.grid_sample(
                self.grid,
                coords,
                mode='bilinear',
                padding_mode='border',
                align_corners=True  # Important for precise alignment
            )
        features = features.squeeze().permute(1, 0)
        # print(features.shape)
        colors = self.mlp(features)
        return colors

    def _create_coords(self, h,w):
        xs, ys = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
        indexes = np.stack((ys.flatten(), xs.flatten()), axis=-1)
        return torch.as_tensor(indexes, dtype=torch.float32)
    def fit(self, img):
        if img.dtype != np.float32: raise TypeError(f"img is not np.float32 != {img.dtype}")

        colors = torch.as_tensor(img, dtype=torch.float32).reshape(-1, OUTPUT_DIM)
        print(f"Using {self.device}")
        indexes = self._create_coords(img.shape[-2], img.shape[-1])
        X = indexes.to(self.device)
        y = colors.to(self.device)
        num_pixels = X.shape[0]
        batch_size = max(64, (2**14)*(self.device == "cuda"))
        steps = math.ceil(num_pixels / batch_size)
        print(f"steps: {steps}")
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        criterion = nn.MSELoss()
        scaler = torch.amp.GradScaler(self.device)
        model.train()
        for epoch in range(100):
            epoch_loss = 0
            perm = torch.randperm(num_pixels, device=self.device)
            for i in range(steps):
                start_idx = i * batch_size
                end_idx = min(start_idx + batch_size, num_pixels)
                indices = perm[start_idx:end_idx]
                batch_X = X[indices].view(1,1,batch_size,2)
                batch_y = y[indices]

                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast(self.device):
                    preds = model(batch_X)
                    loss = criterion(preds, batch_y)
                epoch_loss += loss.item()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            print(epoch_loss)

    def generate(self, h,w):
        coords = self._create_coords(h,w).to(self.device)
        colors = self(coords.view(1,1,-1,2))
        colors = colors.cpu().detach().numpy()
        colors = colors.reshape(h,w)*255
        return colors






img_path = Path("converted/r1u8Y.png")
img = iu.load_image(img_path)
img = iu.float_conv(img)
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = NTC(feature_dim=16, base_res=256, hidden_dim=32,device=dev)
model.fit(img)
gen = model.generate(img.shape[0],img.shape[1])
iu.save_image(img_path.stem+"gen", gen.astype(dtype=np.uint8), colorspace="Y")