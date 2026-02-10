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

from compare import calculate_absolute_difference

def rgb_to_ycocg(img_np):
    """
    Input: (N, 3) RGB in [0, 1]
    Output: (N, 3) YCoCg with Chroma shifted to [0, 1]
    """
    R, G, B = img_np[:, 0], img_np[:, 1], img_np[:, 2]

    Y  =  0.25 * R + 0.5 * G + 0.25 * B
    Co =  0.5  * R           - 0.5  * B
    Cg = -0.25 * R + 0.5 * G - 0.25 * B

    return np.stack([Y, Co + 0.5, Cg + 0.5], axis=1)

def ycocg_to_rgb(img_np):
    """
    Input: (N, 3) YCoCg in [0, 1]
    Output: (N, 3) RGB in [0, 1]
    """
    Y  = img_np[:, 0]
    Co = img_np[:, 1] - 0.5
    Cg = img_np[:, 2] - 0.5

    R = Y + Co - Cg
    G = Y + Cg
    B = Y - Co - Cg

    return np.clip(np.stack([R, G, B], axis=1), 0.0, 1.0)

class LatentFeaturePyramid(nn.Module):
    def __init__(self,
                 feature_dim=4,  

                 base_res=16,  

                 levels=5,  

                 hidden_dim=64):  

        super().__init__()

        self.levels = levels
        self.feature_dim = feature_dim

        self.grids = nn.ParameterList([])
        current_res = base_res

        print(f"Initializing Feature Pyramid with {levels} levels:")
        for i in range(levels):
            print(f"  - Level {i}: {current_res}x{current_res} (Features: {feature_dim})")

            grid = nn.Parameter(
                torch.randn(1, feature_dim, current_res, current_res) * 0.01
            )
            self.grids.append(grid)
            current_res *= 2

        input_dim = levels * feature_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 3),  

            nn.Sigmoid()  

        )

    def forward(self, coords):
        """
        coords: (Batch, 2) in range [0, 1]
        """
        batch_size = coords.shape[0]

        grid_input = (coords * 2 - 1).view(1, 1, batch_size, 2)

        features_list = []

        for grid in self.grids:

            sampled = torch.nn.functional.grid_sample(
                grid,
                grid_input,
                mode='bilinear',
                padding_mode='border',
                align_corners=True  

            )

            features_list.append(sampled.squeeze(2).squeeze(0).t())

        combined_features = torch.cat(features_list, dim=-1)

        rgb = self.mlp(combined_features)

        return rgb

def load_image_data(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found at {path}")

    img = Image.open(path).convert("RGB")
    img_np = np.array(img, dtype=np.float32) / 255.0
    h, w = img_np.shape[:2]

    print("Converting RGB -> YCoCg...")
    colors_rgb = img_np.reshape(-1, 3)
    colors = rgb_to_ycocg(colors_rgb) 

    y = np.linspace(0.5 / h, 1 - 0.5 / h, h)
    x = np.linspace(0.5 / w, 1 - 0.5 / w, w)
    ys, xs = np.meshgrid(y, x, indexing='ij')

    coords = np.stack([ys.flatten(), xs.flatten()], axis=1)

    return coords, colors, h, w

def train_optimized(model, coords, colors, device, epochs=50, batch_size=16384):

    all_coords = torch.from_numpy(coords).float().to(device)
    all_colors = torch.from_numpy(colors).float().to(device)
    num_pixels = all_coords.shape[0]

    try:
        model = torch.compile(model)

    except:
        print("torch.compile not available, skipping.")

    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scaler = torch.amp.GradScaler()
    criterion = nn.MSELoss()

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

def reconstruct(model, device, h, w, scale, file_name):

    model.eval()

    new_h, new_w = int(h * scale), int(w * scale)
    total_pixels = new_h * new_w

    y = np.linspace(0.5 / new_h, 1 - 0.5 / new_h, new_h)
    x = np.linspace(0.5 / new_w, 1 - 0.5 / new_w, new_w)
    ys, xs = np.meshgrid(y, x, indexing='ij')
    coords = torch.from_numpy(np.stack([ys.flatten(), xs.flatten()], axis=1)).float().to(device)

    batch_size = 32768*8
    preds_list = []

    start_time = time.time()
    with torch.no_grad():
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            output = model(chunk)
            preds_list.append(output.cpu().numpy())

    full_img = np.concatenate(preds_list, axis=0)

    print("Converting YCoCg -> RGB...")
    full_img = ycocg_to_rgb(full_img)

    img_reshaped = full_img.reshape(new_h, new_w, 3)
    final_img = (np.clip(img_reshaped, 0, 1) * 255).astype(np.uint8)

    elapsed = time.time() - start_time
    output_path = f"reconstructed/{file_name}_{new_w//1000}k.png"
    Image.fromarray(final_img).save(output_path)

    print(f"  Time: {elapsed:.2f}s | Throughput: {total_pixels / elapsed / 1e6:.2f} MP/s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Latent Feature Pyramid Texture Reconstruction")
    parser.add_argument("--img", type=str, default="data/r8.png", help="Path to input image")
    parser.add_argument("--scale", type=float, default=1.0, help="Output scale factor")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("--out", type=str, default="reconstructed.png", help="Output filename")
    parser.add_argument("--model_path", type=str, default="latent_model2.pth", help="Model save path")
    parser.add_argument("--levels", type=int, default=6, help="Number of pyramid levels")
    parser.add_argument("--feats", type=int, default=4, help="Features per level")
    parser.add_argument("--export", action="store_true", default=True, help="Export weights to JSON for Unity")
    parser.add_argument(
    "--train",
    action="store_true", default=True,
    help="Train a new model instead of loading an existing one"
)

    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device('cuda')
        torch.backends.cudnn.benchmark = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device('cpu')

    try:
        coords, colors, h, w = load_image_data(args.img)
    except Exception as e:
        print(f"Error: {e}")
        exit()

    model = LatentFeaturePyramid(
        base_res=16,
        levels=args.levels,
        feature_dim=args.feats
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model Parameters: {total_params:,}")

    if args.train:
        print("Training new model...")
        train_optimized(model, coords, colors, device, epochs=args.epochs)
        torch.save(model.state_dict(), args.model_path)
        print(f"Model saved to {args.model_path}")
    else:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(
                f"Model file '{args.model_path}' not found. Use --train to create it."
            )
        print(f"Loading model from {args.model_path}")
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    img_path: os.PathLike = pth.Path(args.img)
    file_name = img_path.stem
    for s in [1]:  

        print(f"\nReconstructing at {s}x...")
        reconstruct(model, device, h, w, s, file_name)
    calculate_absolute_difference(file_name[0])

    if args.export:
        import unity_bridge
        print("\n--- Exporting for Unity ---")
        unity_bridge.export_model_for_unity(model, file_name)