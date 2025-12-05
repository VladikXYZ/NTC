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

img = Image.open("aa.png").convert('RGB')
img_array = np.array(img, dtype=np.float32) / 255.0
og, test_indices = smp.image_init(img_array)
all_X, all_y = [], []
for scale in [1, 2, 4, 6]:
    X, y = smp.upsample(og, scale)
    all_X.append(X)
    all_y.append(y)
    # print(f"Scale {scale}: X={X.shape}, y={y.shape}")
X = np.vstack(all_X)  # (total_samples, input_features)
y = np.vstack(all_y)  # (total_samples, 3)

class ColorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 3), nn.Sigmoid()
        )
    def forward(self, x):
        return torch.clamp(self.net(x), 0, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")





full_prec = True
on_gpu = True
halved = False
inputs_tensor = torch.from_numpy(X).float()
targets_tensor = torch.from_numpy(y).float()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 schedule=torch.profiler.schedule(wait=1, warmup=1, active=6),
                 record_shapes=True, profile_memory=True) as prof:

    if on_gpu:
        model = ColorMLP().to(device)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        inputs_tensor = inputs_tensor.to(device)
        targets_tensor = targets_tensor.to(device)
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=1024, shuffle=False, num_workers=0, pin_memory=False)

        # model.train()
        print("Training started")
        start = time.time()


        if full_prec:
            for epoch in range(20):
                epoch_start = time.time()
                for batch_inputs, batch_targets in loader:
                    prof.step()
                    optimizer.zero_grad()
                    preds = model(batch_inputs)
                    loss = criterion(preds, batch_targets)
                    optimizer.step()
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')
        else:
            scaler = GradScaler('cuda')
            for epoch in range(20):
                epoch_start = time.time()
                for batch_inputs, batch_targets in loader:
                    prof.step()
                    optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
                    with autocast('cuda'):
                        preds = model(batch_inputs)
                        loss = criterion(preds, batch_targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')

        print(f"Training took: {time.time() - start:.2f}s")
    else:
        dataset = TensorDataset(inputs_tensor, targets_tensor)
        loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True,
                            persistent_workers=True)

        if full_prec and not halved:
            model = ColorMLP().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scaler = GradScaler('cuda')
            start = time.time()
            for epoch in range(20):
                epoch_start = time.time()
                for batch_inputs, batch_targets in loader:
                    prof.step()
                    batch_inputs = batch_inputs.to(device, non_blocking=True)
                    batch_targets = batch_targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    preds = model(batch_inputs)
                    loss = criterion(preds, batch_targets)
                    # optimizer.step()
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')

        elif halved:
            dataset = TensorDataset(inputs_tensor.half(), targets_tensor.half())
            loader = DataLoader(dataset, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True,
                                persistent_workers=True)
            model = ColorMLP().to(device).half()
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scaler = GradScaler(init_scale=65536)
            start = time.time()
            for epoch in range(20):
                epoch_start = time.time()
                for batch_inputs, batch_targets in loader:
                    prof.step()
                    batch_inputs = batch_inputs.to(device, non_blocking=True)
                    batch_targets = batch_targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    preds = model(batch_inputs)
                    loss = criterion(preds, batch_targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    # optimizer.step()
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')
        else:
            model = ColorMLP().to(device)
            criterion = nn.MSELoss()
            optimizer = optim.Adam(model.parameters(), lr=0.01)
            scaler = GradScaler('cuda')
            start = time.time()
            for epoch in range(20):
                epoch_start = time.time()
                for batch_inputs, batch_targets in loader:
                    prof.step()
                    batch_inputs = batch_inputs.to(device, non_blocking=True)
                    batch_targets = batch_targets.to(device, non_blocking=True)
                    optimizer.zero_grad(set_to_none=True)
                    with autocast('cuda'):
                        preds = model(batch_inputs)
                        loss = criterion(preds, batch_targets)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')
        print(f"Training took: {time.time() - start:.2f}s")
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
h, w = og.shape[:2]
og_np = og.cpu().numpy() if torch.is_tensor(og) else og
smp.reconstruct_image(model, device, h, w, halved=halved)
smp.reconstruct_image(model, device, h, w, 32, halved=halved)

# Image.fromarray(predicted_colors).save(f"reconstructed{z}x.png")
# print(f"✓ Reconstructed image with {z} zoom saved: reconstructed{z}x.png")
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(og_np); axes[0].set_title('Original'); axes[0].axis('off')
# axes[1].imshow(predicted_colors); axes[1].set_title('MLP Output'); axes[1].axis('off')
# diff = np.mean(np.abs(predicted_colors - og_np), axis=2)
# im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
# axes[2].set_title('Difference'); axes[2].axis('off')
# plt.tight_layout()
# plt.show()
