import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from tqdm import tqdm
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

INPUT_IMAGE = "Input/texture2.jpg"
if not os.path.exists("Output"):
    os.makedirs("Output")

CONFIG = {
    'grid_size': 512,
    'grid_channels': 8,
    'noise_channels': 32,
    'octaves': 4,
    'base_scale': 20.0,
    'hidden_dim': 64,
    'lr': 0.005
}

img = Image.open(INPUT_IMAGE).convert("RGB")
target = ToTensor()(img).unsqueeze(0).to(device)
_, _, H, W = target.shape
print(f"Training resolution: {W}x{H}")

class MultiOctaveNoiseGenerator(nn.Module):
    def __init__(self, total_channels, octaves, base_scale):
        super().__init__()
        self.octaves = octaves
        self.channels_per_octave = total_channels // octaves
        self.freq_layers = nn.ParameterList([])

        for i in range(octaves):
            current_scale = base_scale * (2 ** i)
            w_shape = (self.channels_per_octave // 2, 2)

            weights = torch.randn(w_shape) * current_scale
            self.freq_layers.append(nn.Parameter(weights, requires_grad=False))

    def forward(self, h, w, y_range=(0,1), x_range=(0,1)):
        y = torch.linspace(y_range[0], y_range[1], h, device=device)
        x = torch.linspace(x_range[0], x_range[1], w, device=device)

        yy, xx = torch.meshgrid(y, x, indexing='ij')
        coords = torch.stack([xx, yy], dim=-1) 

        noise_outputs = []
        for i in range(self.octaves):
            freq_weights = self.freq_layers[i] 
            projection = coords @ freq_weights.T
            layer_noise = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
            noise_outputs.append(layer_noise)

        full_noise = torch.cat(noise_outputs, dim=-1)
        return full_noise.permute(2, 0, 1).unsqueeze(0)

class NeuralDetailMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config 

        self.grid = nn.Parameter(torch.randn(1, config['grid_channels'], config['grid_size'], config['grid_size']) * 0.1)
        self.noise_gen = MultiOctaveNoiseGenerator(config['noise_channels'], config['octaves'], config['base_scale'])

        input_dim = config['grid_channels'] + config['noise_channels']
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, config['hidden_dim'], 1),
            nn.ReLU(),
            nn.Conv2d(config['hidden_dim'], config['hidden_dim'], 1),
            nn.ReLU(),
            nn.Conv2d(config['hidden_dim'], 3, 1), 
            nn.Sigmoid()
        )

    def get_features(self, h, w, y_range=(0,1), x_range=(0,1)):

        v_noise = self.noise_gen(h, w, y_range, x_range)

        y_min, y_max = y_range[0] * 2 - 1, y_range[1] * 2 - 1
        x_min, x_max = x_range[0] * 2 - 1, x_range[1] * 2 - 1

        grid_y = torch.linspace(y_min, y_max, h, device=device)
        grid_x = torch.linspace(x_min, x_max, w, device=device)
        gy, gx = torch.meshgrid(grid_y, grid_x, indexing='ij')

        sample_coords = torch.stack([gx, gy], dim=-1).unsqueeze(0)

        v_grid = torch.nn.functional.grid_sample(
            self.grid, 
            sample_coords, 
            mode='bilinear', 
            align_corners=False
        )

        return torch.cat([v_grid, v_noise], dim=1)

    def forward(self):

        features = self.get_features(H, W, (0,1), (0,1))
        return self.mlp(features)

    def render_tile(self, out_h, out_w, y_range, x_range):
        features = self.get_features(out_h, out_w, y_range, x_range)
        return self.mlp(features)

model = NeuralDetailMapper(CONFIG).to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
loss_fn = nn.MSELoss()
scaler = torch.GradScaler('cuda')

print(f"Training with {CONFIG['octaves']} octaves...")

train_x = None
train_y = target
val_x = None
val_y = target

best_val = float('inf')
no_improve = 0
patience = 200
max_steps = 1000

pbar = tqdm(total=max_steps, desc="Learning Texture")

for step in range(max_steps):
    optimizer.zero_grad()
    with torch.autocast('cuda'):
        out = model()
        loss = loss_fn(out, train_y)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    with torch.no_grad():
        with torch.autocast('cuda'):
            val_pred = model()
            val_loss = loss_fn(val_pred, val_y).item()

    pbar.set_description(f"Train {loss.item():.5f} | Val {val_loss:.5f}")
    pbar.update(1)

    if val_loss < best_val:
        best_val = val_loss
        no_improve = 0
    else:
        no_improve += 1

    if no_improve >= patience:
        break

pbar.close()

print("\nSaving model checkpoint...")
checkpoint_path = "Output/saved_model.pth"

torch.save({
    'model_state_dict': model.state_dict(), 

    'config': CONFIG,                       

    'training_resolution': (H, W)           

}, checkpoint_path)

print(f"Model saved to {checkpoint_path}")

def render_tiled(model, total_h, total_w, tile_size=512):
    """
    Renders the image in small squares (tiles) to save VRAM.
    """
    print(f"Rendering {total_w}x{total_h} in {tile_size}x{tile_size} tiles...")

    full_image = Image.new('RGB', (total_w, total_h))

    y_steps = range(0, total_h, tile_size)
    x_steps = range(0, total_w, tile_size)

    total_tiles = len(y_steps) * len(x_steps)
    pbar = tqdm(total=total_tiles, desc="Stitching tiles")

    model.eval()
    with torch.no_grad():
        for y in y_steps:
            for x in x_steps:

                h_curr = min(tile_size, total_h - y)
                w_curr = min(tile_size, total_w - x)

                y_start = y / total_h
                y_end = (y + h_curr) / total_h
                x_start = x / total_w
                x_end = (x + w_curr) / total_w

                chunk_tensor = model.render_tile(
                    h_curr, 
                    w_curr, 
                    y_range=(y_start, y_end), 
                    x_range=(x_start, x_end)
                )

                chunk_img = ToPILImage()(chunk_tensor[0].clamp(0, 1).cpu())

                full_image.paste(chunk_img, (x, y))

                pbar.update(1)

                del chunk_tensor

    pbar.close()
    return full_image

checkpoint = torch.load(checkpoint_path, map_location=device)
loaded_model = NeuralDetailMapper(checkpoint['config']).to(device)
loaded_model.load_state_dict(checkpoint['model_state_dict'])

TARGET_H = H * 2
TARGET_W = W * 2

final_image = render_tiled(loaded_model, TARGET_H * 2, TARGET_W * 2, tile_size=512)

final_image = final_image.resize((TARGET_W, TARGET_H), resample=Image.LANCZOS)

save_path = f"Output/output_tiled_{TARGET_W}x{TARGET_H}.png"
final_image.save(save_path)
print(f"Saved high-res image to {save_path}")
print("Done.")

import struct

print("\n--- FIXED UNITY EXPORT ---")
output_dir = "Output/UnityData"
if not os.path.exists(output_dir): os.makedirs(output_dir)

def save_tensor_safe(tensor, filename):

    t = tensor.detach().cpu()

    t = t.to(torch.float32)

    flat = t.numpy().flatten(order='C')

    flat.tofile(filename)
    print(f"Saved {filename} | Shape: {tuple(tensor.shape)} | First Val: {flat[0]:.6f}")

grid_full = model.grid[0].permute(1, 2, 0).detach().cpu() 

grid_a = grid_full[:, :, 0:4] 

grid_b = grid_full[:, :, 4:8] 

save_tensor_safe(grid_a, f"{output_dir}/grid_texture_a.bytes")
save_tensor_safe(grid_b, f"{output_dir}/grid_texture_b.bytes")

all_freqs_list = [layer.detach().cpu().flatten() for layer in model.noise_gen.freq_layers]
all_freqs_tensor = torch.cat(all_freqs_list)
save_tensor_safe(all_freqs_tensor, f"{output_dir}/noise_freqs.bytes")

mlp = model.mlp

save_tensor_safe(mlp[0].weight.reshape(64, 40), f"{output_dir}/l0_weight.bytes")
save_tensor_safe(mlp[0].bias, f"{output_dir}/l0_bias.bytes")

save_tensor_safe(mlp[2].weight.reshape(64, 64), f"{output_dir}/l1_weight.bytes")
save_tensor_safe(mlp[2].bias, f"{output_dir}/l1_bias.bytes")

save_tensor_safe(mlp[4].weight.reshape(3, 64), f"{output_dir}/l2_weight.bytes")
save_tensor_safe(mlp[4].bias, f"{output_dir}/l2_bias.bytes")

print("Export Done. Copy 'UnityData' to Unity project.")

print("\n--- GENERATING DEBUG IMAGES ---")

def save_normalized_debug(tensor, filename):
    """
    Normalizuje tensor do rozsahu 0-1 a uloží jako PNG.
    Důležité, protože raw hodnoty mohou být mimo rozsah barev.
    """

    if tensor.shape[0] > 3:
        viz_tensor = tensor[0:3].clone()
    else:
        viz_tensor = tensor.clone()

    t_min, t_max = viz_tensor.min(), viz_tensor.max()
    print(f"Exporting {filename} | Range: {t_min:.4f} to {t_max:.4f}")

    if t_max - t_min > 1e-6:
        viz_tensor = (viz_tensor - t_min) / (t_max - t_min)
    else:
        viz_tensor = viz_tensor - t_min 

    ToPILImage()(viz_tensor.clamp(0, 1)).save(filename)

with torch.no_grad():

    raw_grid = model.grid[0].detach().cpu()
    save_normalized_debug(raw_grid, "Output/debug_python_grid_rgb.png")

    h_debug, w_debug = 512, 512

    raw_noise = model.noise_gen(h_debug, w_debug)[0].detach().cpu() 

    save_normalized_debug(raw_noise, "Output/debug_python_noise_rgb.png")

    noise_sum = raw_noise.sum(dim=0, keepdim=True) 

    save_normalized_debug(noise_sum, "Output/debug_python_noise_sum.png")

print("Debug images saved to 'Output/'. Compare them with Unity!")

print("\n--- VALIDATION VALUES (Check these against Unity Console) ---")
print(f"Noise Freqs [0]: {model.noise_gen.freq_layers[0].flatten()[0].item():.6f}")

print(f"Layer0 Weight [0]: {model.mlp[0].weight.flatten()[0].item():.6f}")
print("-------------------------------------------------------------")