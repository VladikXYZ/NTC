import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch import GradScaler, autocast
from torch.utils.data import TensorDataset, DataLoader
import sampling as smp
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from torch.profiler import profile, ProfilerActivity


if __name__ == "__main__":
    img = Image.open("aa.png").convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    og, test_indices = smp.image_init(img_array)
    og = img_array
    all_X, all_y = [], []
    for scale in [0.5, 1, 2, 4, 6]:
        X, y = smp.upsample(og, scale)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.vstack(all_y)

    class ColorMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(2, 128), nn.Sigmoid(),
                nn.Linear(128, 64), nn.Sigmoid(),
                nn.Linear(64, 3), nn.Sigmoid()
            )
        def forward(self, x):
            return self.net(x)

    inputs_tensor = torch.from_numpy(X).float()
    targets_tensor = torch.from_numpy(y).float()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #                  schedule=torch.profiler.schedule(wait=1, warmup=1, active=6),
    #                  record_shapes=True, profile_memory=True) as prof:
    dataset = TensorDataset(inputs_tensor, targets_tensor)
    loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0)
    model = ColorMLP()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    print("Training starts")
    start = time.time()
    for epoch in range(100):
        epoch_start = time.time()
        for batch_inputs, batch_targets in loader:
            # prof.step()
            batch_inputs = batch_inputs
            batch_targets = batch_targets
            optimizer.zero_grad()
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    h, w = og.shape[:2]
    smp.reconstruct_image(model, "cpu", h, w, bench=False)
    smp.reconstruct_image(model, "cpu", h, w, 4, bench=False)