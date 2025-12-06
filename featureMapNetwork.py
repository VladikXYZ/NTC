import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from tqdm import tqdm
import os

# --- CONSTANTS ---
GRID_SIZE = 128
GRID_CHANNELS = 8
NOISE_CHANNELS = 32
OCTAVES = 4
BASE_SCALE = 20.0
HIDDEN_DIM = 64
LEARNING_RATE = 0.005
PATIENCE = 200
INPUT_IMAGE = "data/aa.png"
CHECKPOINT_PATH = "lama_model.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)

img = Image.open(INPUT_IMAGE).convert("RGB")
target = ToTensor()(img).unsqueeze(0).to(device)
_, _, H, W = target.shape
print(f"Training resolution: {W}x{H}")


# --- 1. NOISE GENERATOR ---
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

    def forward(self, h, w, y_range=(0, 1), x_range=(0, 1)):
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


# --- 2. NEURAL DETAIL MAPPER ---
class NeuralDetailMapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.grid = nn.Parameter(torch.randn(1, GRID_CHANNELS, GRID_SIZE, GRID_SIZE) * 0.1)
        self.noise_gen = MultiOctaveNoiseGenerator(NOISE_CHANNELS, OCTAVES, BASE_SCALE)

        input_dim = GRID_CHANNELS + NOISE_CHANNELS
        self.mlp = nn.Sequential(
            nn.Conv2d(input_dim, HIDDEN_DIM, 1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_DIM, HIDDEN_DIM, 1),
            nn.ReLU(),
            nn.Conv2d(HIDDEN_DIM, 3, 1),
            nn.Sigmoid()
        )

    def get_features(self, h, w, y_range=(0, 1), x_range=(0, 1)):
        # Branch B: Noise
        v_noise = self.noise_gen(h, w, y_range, x_range)

        # Branch A: Grid
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
        features = self.get_features(H, W, (0, 1), (0, 1))
        return self.mlp(features)

    def render_tile(self, out_h, out_w, y_range, x_range):
        features = self.get_features(out_h, out_w, y_range, x_range)
        return self.mlp(features)


# Initialize model
model = NeuralDetailMapper().to(device)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
loss_fn = nn.MSELoss()
scaler = torch.GradScaler('cuda')

# --- TRAINING ---
print(f"Training with {OCTAVES} octaves...")
best_loss = float("inf")
no_improve = 0

pbar = tqdm(total=PATIENCE, desc="Learning Texture")
while True:
    optimizer.zero_grad()

    with torch.autocast('cuda'):
        out = model()
        loss = loss_fn(out, target)

    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    loss_val = loss.item()
    pbar.set_description(f"Loss {loss_val:.5f}")
    pbar.update(0)

    if loss_val < best_loss:
        best_loss = loss_val
        no_improve = 0
        pbar.n = 0
        pbar.refresh()
    else:
        no_improve += 1
        pbar.n = no_improve
        pbar.refresh()

    if no_improve >= PATIENCE:
        break

pbar.close()

# --- SAVE MODEL ---
print("\nSaving model checkpoint...")
torch.save({
    'model_state_dict': model.state_dict(),
    'training_resolution': (H, W)
}, CHECKPOINT_PATH)
print(f"Model saved to {CHECKPOINT_PATH}")


# --- LOAD AND RENDER ---
def load_trained_model(path):
    print(f"Loading from {path}...")
    checkpoint = torch.load(path, map_location=device)

    new_model = NeuralDetailMapper().to(device)
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.eval()

    print("Model loaded successfully!")
    return new_model


loaded_model = load_trained_model(CHECKPOINT_PATH)

with torch.no_grad():
    print("Rendering native resolution from loaded model...")
    recon = loaded_model.render_tile(H, W, (0, 1), (0, 1))[0].detach().cpu()
    ToPILImage()(recon).save("output_loaded_native.png")
    print("Saved output_loaded_native.png")

print("Done.")
