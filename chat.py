import os
import argparse
import time
import math
import numpy as np
import pathlib as pth
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from PIL import Image


# ==========================================
# 1. SOTA ARCHITECTURE: FEATURE PYRAMID
# ==========================================

class LatentFeaturePyramid(nn.Module):
    def __init__(self,
                 feature_dim=4,  # Features per grid level
                 base_res=16,  # Lowest resolution
                 levels=4,  # Number of levels (32, 64, 128, 256, 512, 1024)
                 hidden_dim=64):  # MLP hidden size
        super().__init__()

        self.levels = levels
        self.feature_dim = feature_dim

        # 1. Create a Pyramid of Grids
        # Instead of one grid, we use multiple at geometric resolutions.
        # This provides "SOTA" quality by separating low/high freq details
        # without using hash collisions or random noise.
        self.grids = nn.ParameterList([])
        current_res = base_res

        print(f"Initializing Feature Pyramid with {levels} levels:")
        for i in range(levels):
            print(f"  - Level {i}: {current_res}x{current_res} (Features: {feature_dim})")
            # Initialize with small random noise for training stability
            grid = nn.Parameter(
                torch.randn(1, feature_dim, current_res, current_res) * 0.01
            )
            self.grids.append(grid)
            current_res *= 2

        # 2. MLP Decoder
        # Input size is sum of all level features
        input_dim = levels * feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  # RGB Output
            nn.Sigmoid()  # Force [0, 1] range
        )

    def forward(self, coords):
        """
        coords: (Batch, 2) in range [0, 1]
        """
        batch_size = coords.shape[0]

        # 1. Prepare coordinates for grid_sample
        # grid_sample expects [-1, 1], we have [0, 1]
        # (N, 1, 1, 2) format required for 4D input
        grid_input = (coords * 2 - 1).view(1, 1, batch_size, 2)

        features_list = []

        # 2. Sample from every level of the pyramid
        for grid in self.grids:
            # Result: (1, Feature_Dim, 1, Batch)
            sampled = torch.nn.functional.grid_sample(
                grid,
                grid_input,
                mode='bilinear',
                padding_mode='border',
                align_corners=True  # Important for precise alignment
            )
            # Reshape to (Batch, Feature_Dim)
            features_list.append(sampled.squeeze(2).squeeze(0).t())

        # 3. Concatenate all levels
        # Shape: (Batch, Levels * Feature_Dim)
        combined_features = torch.cat(features_list, dim=-1)

        # 4. Decode to RGB
        rgb = self.mlp(combined_features)

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

    print(f"Loaded Image: {w}x{h}")

    # Generate pixel center coordinates
    # We want coordinates to correspond exactly to pixel centers
    y = np.linspace(0.5 / h, 1 - 0.5 / h, h)
    x = np.linspace(0.5 / w, 1 - 0.5 / w, w)
    ys, xs = np.meshgrid(y, x, indexing='ij')

    coords = np.stack([ys.flatten(), xs.flatten()], axis=1)  # (N, 2)
    colors = img_np.reshape(-1, 3)  # (N, 3)

    return coords, colors, h, w


# ==========================================
# 3. TRAINING LOOP
# ==========================================

def train_optimized(model, coords, colors, device, epochs=50, batch_size=65536):
    print(f"\n--- Starting Optimized Training on {device} ---")
    print("Moving dataset to VRAM...")
    all_coords = torch.from_numpy(coords).float().to(device)
    all_colors = torch.from_numpy(colors).float().to(device)
    num_pixels = all_coords.shape[0]

    try:
        model = torch.compile(model)
        print("Model compiled with torch.compile() ✓")
    except:
        print("torch.compile not available, skipping.")

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler()
    criterion = nn.MSELoss()

    # Calculate iterations per epoch based on batch size
    # We want to cover roughly the whole image per epoch
    steps_per_epoch = math.ceil(num_pixels / batch_size)

    model.train()
    start_time = time.time()

    for epoch in range(epochs):
        epoch_loss = 0

        for _ in range(steps_per_epoch):
            indices = torch.randint(0, num_pixels, (batch_size,), device=device)
            batch_X = all_coords[indices]
            batch_y = all_colors[indices]

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda"):
                preds = model(batch_X)
                loss = criterion(preds, batch_y)

            # Scale gradients and step
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        # Logging
        if (epoch + 1) % 5 == 0 or epoch == 0:
            avg_loss = epoch_loss / steps_per_epoch
            psnr = -10 * math.log10(avg_loss) if avg_loss > 0 else 100
            print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {avg_loss:.6f} | PSNR: {psnr:.2f}dB")

    total_time = time.time() - start_time
    print(f"Training finished in {total_time:.2f}s")


# ==========================================
# 4. RECONSTRUCTION
# ==========================================

def reconstruct(model, device, h, w, scale, file_name):
    print(f"\n--- Reconstructing at Scale {scale}x ---")
    model.eval()

    new_h, new_w = int(h * scale), int(w * scale)
    total_pixels = new_h * new_w
    print(f"Target Resolution: {new_w}x{new_h}")

    # Dense coordinates for inference
    y = np.linspace(0.5 / new_h, 1 - 0.5 / new_h, new_h)
    x = np.linspace(0.5 / new_w, 1 - 0.5 / new_w, new_w)
    ys, xs = np.meshgrid(y, x, indexing='ij')
    coords = torch.from_numpy(np.stack([ys.flatten(), xs.flatten()], axis=1)).float().to(device)
    # Batched Inference to handle any resolution on 3060
    batch_size = 32768*8
    preds_list = []

    # Use torch.no_grad and stream to CPU to save VRAM
    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            output = model(chunk)
            preds_list.append(output.cpu().numpy())

    full_img = np.concatenate(preds_list, axis=0)
    img_reshaped = full_img.reshape(new_h, new_w, 3)

    # Save
    elapsed = time.time() - start_time
    final_img = (np.clip(img_reshaped, 0, 1) * 255).astype(np.uint8)
    output_path = f"reconstructed/{file_name}_{scale}x.png"
    Image.fromarray(final_img).save(output_path)
    print(f"✓ Saved to {output_path}")
    print(f"  Time: {elapsed:.2f}s | Throughput: {total_pixels / elapsed / 1e6:.2f} MP/s")


# ==========================================
# 5. MAIN
# ==========================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Feature Pyramid Texture Reconstruction")
    parser.add_argument("--img", type=str, default="data/rw1.png", help="Path to input image")
    parser.add_argument("--scale", type=float, default=2.0, help="Output scale factor")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--out", type=str, default="reconstructed.png", help="Output filename")
    parser.add_argument("--model_path", type=str, default="latent_model.pth", help="Model save path")
    parser.add_argument("--levels", type=int, default=6, help="Number of pyramid levels")
    parser.add_argument("--feats", type=int, default=4, help="Features per level")

    args = parser.parse_args()

    # Device Config
    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    # 1. Load
    try:
        coords, colors, h, w = load_image_data(args.img)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    model = LatentFeaturePyramid(base_res=16, levels=6).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")
    train_optimized(model, coords, colors, device, epochs=args.epochs)
    torch.save(model.state_dict(), args.model_path)
    print(f"Model saved: {args.model_path}")

    img_path: os.PathLike = pth.Path(args.img)
    file_name = img_path.stem
    for s in [1]:
        reconstruct(model, device, h, w, s, file_name)