import os
import argparse
import time
import math
import numpy as np
import pathlib as pth
import torch
import torch.nn as nn
import torch.optim as optim
from sympy import true
from torch._C._profiler import ProfilerActivity
from torch.profiler import profile
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

# ==========================================
# 1. SOTA ARCHITECTURE: FEATURE PYRAMID
# ==========================================

class LatentFeaturePyramid(nn.Module):
    def __init__(self, feature_dim=4, base_res=16, hidden_dim=64):
        super().__init__()
        self.feature_dim = feature_dim

        # print(f"Initializing Feature Pyramid with {levels} levels:")
        # for i in range(levels):
        #     print(f"  - Level {i}: {current_res}x{current_res} (Features: {feature_dim})")
        #     grid = nn.Parameter(torch.randn(1, feature_dim, current_res, current_res) * 0.01)
        #     self.grids.append(grid)
        #     # current_res *= 2

        print(f"  - Levels: {base_res}x{base_res} (Features: {feature_dim})")
        self.grid = nn.Parameter(torch.randn(1, feature_dim, base_res, base_res) * 0.01)

        input_dim = feature_dim
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid() )

    def forward(self, coords):
        batch_size = coords.shape[0]
        grid_input = (coords * 2 - 1).view(1, 1, batch_size, 2)

        # features_list = []
        # for grid in self.grids:
        #     # Result: (1, Feature_Dim, 1, Batch)
        #     sampled = torch.nn.functional.grid_sample(
        #         grid,
        #         grid_input,
        #         mode='bilinear',
        #         padding_mode='border',
        #         align_corners=True  # Important for precise alignment
        #     )
        #     # Reshape to (Batch, Feature_Dim)
        #     features_list.append(sampled.squeeze(2).squeeze(0).t())

        features =  torch.nn.functional.grid_sample(
                self.grid,
                grid_input,
                mode='bilinear',
                padding_mode='border',
                align_corners=True  # Important for precise alignment
            )
        features = features.squeeze(2).squeeze(0).permute(1, 0)
        rgb = self.mlp(features)
        return rgb


# ==========================================
# 2. UTILS & PREPROCESSING
# ==========================================

def load_image_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    h, w = img_np.shape[:2]

    y = np.linspace(0.5 / h, 1 - 0.5 / h, h)
    x = np.linspace(0.5 / w, 1 - 0.5 / w, w)
    ys, xs = np.meshgrid(y, x, indexing='ij')

    coords = np.stack([ys.flatten(), xs.flatten()], axis=1)
    colors = img_np.reshape(-1, 3)

    return coords, colors, h, w


# ==========================================
# 3. TRAINING LOOP
# ==========================================

def train_optimized(model, coords, colors, device, epochs=50, batch_size=65536):
    all_coords = torch.from_numpy(coords).float().to(device)
    all_colors = torch.from_numpy(colors).float().to(device)
    num_pixels = all_coords.shape[0]

    try:
        model = torch.compile(model)
    except:
        print("torch.compile not available, skipping.")

    if device == "cpu": batch_size = 256
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler(str(device))
    criterion = nn.MSELoss()
    steps_per_epoch = math.ceil(num_pixels / batch_size)
    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0
        perm = torch.randperm(num_pixels, device=device)
        for i in range(steps_per_epoch):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, num_pixels)
            indices = perm[start_idx:end_idx]
            batch_X = all_coords[indices]
            batch_y = all_colors[indices]

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(str(device)):
                preds = model(batch_X)
                loss = criterion(preds, batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()

        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / steps_per_epoch
            psnr = -10 * math.log10(avg_loss) if avg_loss > 0 else 100
            print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {avg_loss:.6f} | PSNR: {psnr:.2f}dB")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")


# ==========================================
# 4. RECONSTRUCTION
# ==========================================
class Reconstruct:

    def multi_reconstruct(self, model, device, h, w, scales, file_name, save=False):
        max_scale = max(scales)
        # model = torch.compile(model, mode="max-autotune")
        model.eval()
        new_h, new_w = int(h * max_scale), int(w * max_scale)
        total_pixels = new_h * new_w
        self.output_fp = torch.zeros((total_pixels, 3)).to(device)
        self.output_ui = torch.zeros((total_pixels, 3)).to(torch.uint8).to(device)
        self.save = save
        for scale in scales:
            new_h, new_w = int(h * scale), int(w * scale)
            y = np.linspace(0.5 / new_h, 1 - 0.5 / new_h, new_h)
            x = np.linspace(0.5 / new_w, 1 - 0.5 / new_w, new_w)
            ys, xs = np.meshgrid(y, x, indexing='ij')
            coords = torch.from_numpy(np.stack([ys.flatten(), xs.flatten()], axis=1)).float().to(device)
            self.reconstruct(model, coords, new_h, new_w,  file_name)


    def reconstruct(self, model, coords, h, w,  file_name):
        batch_size = 32768*256
        total = w*h
        # start_time = time.time()

        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float16):
                    for i in range(0, total, batch_size):
                        idx1, idx2 = i, min(total, i+batch_size)
                        chunk = coords[idx1:idx2]
                        self.output_fp[idx1:idx2] = model(chunk)

            self.output_ui = (self.output_fp*255).to(torch.uint8)
            # elapsed = time.time() - start_time
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))
        if self.save:
            output_path = f"reconstructed/{file_name}_{w // 1000}k.png"
            to_save = self.output_ui[:total].reshape(h, w, 3)
            final_img = to_save.cpu().numpy()
            Image.fromarray(final_img).save(output_path)

        # final_img = self.output_ui[:total].reshape(h, w, 3).cpu().numpy()
        # output_path = f"reconstructed/{file_name}_{w//1000}k.png"
        # Image.fromarray(final_img).save(output_path)

        # print(f" Time: {elapsed:.4f}s | Throughput: {coords.shape[0] / elapsed / 1e6:.2f} MP/s | Reconstructed {h}x{w}")

# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    train = True
    save = True
    base = 512
    feature_dim = 4
    epochs = 1000
    path = "data/r8.png"
    img_path: os.PathLike = pth.Path(path)
    file_name = img_path.stem
    model_path = f"models/{file_name}.pth"

    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    try:
        coords, colors, h, w = load_image_data(path)
    except Exception as e:
        print(f"Error: {e}")
        exit()


    if train:
        model = LatentFeaturePyramid(base_res=base, feature_dim=feature_dim).to(device)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Model Parameters: {total_params:,}")
        train_optimized(model, coords, colors, device, epochs=epochs)
        torch.save(model.state_dict(), model_path)
    else:
        model = LatentFeaturePyramid(base_res=base, feature_dim=feature_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        rec = Reconstruct()
        scales = [0.125, 0.125, 0.25, 0.5, 1, 1.25]
        rec.multi_reconstruct(model, device, h, w, scales, file_name, save)

    # # Export?
    # if args.export:
    #     import unity_bridge
    #     print("\n--- Exporting for Unity ---")
    #     unity_bridge.export_model_for_unity(model, file_name)