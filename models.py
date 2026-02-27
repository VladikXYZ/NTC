import math
import time
from pathlib import Path

import numpy as np

import image_utils as iu
import torch


COLOR_SPACE = "Y"
OUTPUT_DIMS = {"RGB": 3, "YCoCg": 3, "Y":1}
OUTPUT_DIM = OUTPUT_DIMS[COLOR_SPACE]
HIDDEN_DIM = 32
FEATURE_DIM = 16
FEATURES_CHANNEL = 4
POSITIONAL_ENCODING = 0
BASE_RES = 1024
EPOCHS = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = max(4096, (2**18)*(DEVICE=="cuda"))

def _create_coords(h, w):
    xs, ys = np.meshgrid(np.linspace(-1, 1, h), np.linspace(-1, 1, w))
    indexes = np.stack((ys.flatten(), xs.flatten()), axis=-1)
    return torch.as_tensor(indexes, dtype=torch.float32)

def _create_mlp():
    return torch.nn.Sequential(
        torch.nn.Linear(FEATURE_DIM, HIDDEN_DIM),torch.nn.ReLU(inplace=True),
        torch.nn.Linear(HIDDEN_DIM, HIDDEN_DIM), torch.nn.ReLU(inplace=True),
        torch.nn.Linear(HIDDEN_DIM, OUTPUT_DIM),torch.nn.Tanh())

def _create_positional_encoding(coords):
    freq_bands = (2.0 ** torch.arange(POSITIONAL_ENCODING, device=coords.device)) * math.pi
    scaled = coords.unsqueeze(-1) * freq_bands
    encoded = torch.cat([torch.sin(scaled), torch.cos(scaled)], dim=-1)
    res = encoded.view(coords.shape[0], -1)
    return res

def _fit(model,img):
    if img.dtype != torch.float32: raise TypeError(f"img must be np.float32 not {img.dtype}")
    colors = img.reshape(-1, OUTPUT_DIM)
    indexes = _create_coords(img.shape[0], img.shape[1])
    X = indexes.to(DEVICE)
    y = colors.to(DEVICE)
    num_pixels = X.shape[0]

    # if DEVICE == "cuda":
    #     try: model = torch.compile(model)
    #     except: print("torch.compile not available, skipping.")

    steps = math.ceil(num_pixels / BATCH_SIZE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    scaler = torch.amp.GradScaler(DEVICE)
    model.train()
    print(f"Training on {DEVICE} started ...")
    print(f"Epochs: {EPOCHS}\nData size: {num_pixels:,}\nBatch size: {BATCH_SIZE:,}\nSteps: {steps:,}")
    start = time.time()
    epoch_loss = torch.zeros(1).to(DEVICE)
    for epoch in range(EPOCHS):
        ep_start = time.time()
        epoch_loss*=0
        perm = torch.randperm(num_pixels, device=DEVICE)
        start_idx = 0
        for i in range(steps):
            end_idx = min(start_idx + BATCH_SIZE, num_pixels)
            indices = perm[start_idx:end_idx]
            batch_X = X[indices]
            batch_y = y[indices]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(DEVICE):
                preds = model(batch_X)
                batch_loss = ((preds - batch_y).pow(2)).sum()
                loss = batch_loss/(end_idx-start_idx)
                epoch_loss += batch_loss

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            start_idx += BATCH_SIZE
        epoch_loss = epoch_loss.item()/num_pixels
        psnr = -20 * np.log10(np.sqrt(epoch_loss))
        print(f"Epoch {epoch:3d}, Loss: {epoch_loss:.7f}, PSNR: {psnr:.2f}, Time: {time.time() - ep_start:.3f}")
    print(f"Training done in {time.time() - start:.3f} seconds")


def _generate(model, h,w):
    num_pixels = h*w
    model.eval()
    coords = _create_coords(h,w).to(DEVICE)
    steps = math.ceil(num_pixels / BATCH_SIZE)
    colors = torch.zeros((num_pixels, OUTPUT_DIM))
    with torch.no_grad():
        for i in range(steps):
            start_idx = i * BATCH_SIZE
            end_idx = min(start_idx + BATCH_SIZE, num_pixels)
            colors[start_idx:end_idx] = model(coords[start_idx:end_idx])
    colors = colors.reshape((h,w, OUTPUT_DIM)).squeeze().cpu().detach().numpy()
    colors = (colors*255).astype(np.uint8)
    return colors

class PNTC(torch.nn.Module):
    def __init__(self):
        super().__init__()
        levels = FEATURE_DIM//FEATURES_CHANNEL
        self.grids = torch.nn.ParameterList([])
        current_res = BASE_RES >> (levels-1)
        print(f"Initializing Feature Pyramid with {levels} levels:")
        for i in range(levels):
            print(f"  - Level {i}: {current_res}x{current_res} (Features: {FEATURES_CHANNEL})")
            grid = torch.nn.Parameter(torch.randn(1, FEATURES_CHANNEL, current_res, current_res) * 0.01)
            self.grids.append(grid)
            current_res *= 2
        self.mlp = _create_mlp()
        self.to(DEVICE)

    def forward(self, coords):
        grid_input = coords.view(1, 1, -1, 2)
        features_list = []
        for grid in self.grids:
            sampled = torch.nn.functional.grid_sample(
                grid, grid_input, mode='bilinear',
                padding_mode='border', align_corners=True)
            features_list.append(sampled.squeeze().t())
        combined_features = torch.cat(features_list, dim=-1)
        rgb = self.mlp(combined_features)
        return rgb

img_path = Path(f"converted/r4_u8{COLOR_SPACE}.png")
tensor = iu.tensor_float_load(img_path)
h,w = tensor.shape[:2]
dev = "cuda" if torch.cuda.is_available() else "cpu"
model = PNTC()
_fit(model, tensor)
for size in [1024,2048,4096]:
    gen = _generate(model, size, size)
    iu.save_image(img_path.stem[0]+str(size//1024), gen, COLOR_SPACE, True)