import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import preprocessing as smp


GRID_SIZE = 128
GRID_CHANNELS = 8
NOISE_CHANNELS = 32
OCTAVES = 4
BASE_SCALE = 20.0
HIDDEN_DIM = 64
LEARNING_RATE = 0.005
PATIENCE = 200

if __name__ == "__main__":

    # Load image
    img = Image.open("data/oa.png").convert("RGB")
    img_array = np.array(img, dtype=np.float32) / 255.0
    print(img_array.shape)
    H, W = img_array.shape[:2]
    print(f"Training resolution: {W}x{H}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Generate coordinate grid dataset
    y = torch.linspace(0, 1, H)
    x = torch.linspace(0, 1, W)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)
    colors = img_array.reshape(-1, 3)


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

        def forward(self, coords):
            """
            Args:
                coords: (batch_size, 2) tensor with [x, y] coordinates in [0, 1]
            Returns:
                (batch_size, NOISE_CHANNELS) noise features
            """
            noise_outputs = []
            for i in range(self.octaves):
                freq_weights = self.freq_layers[i]
                # (batch_size, 2) @ (2, channels_per_octave // 2).T = (batch_size, channels_per_octave // 2)
                projection = coords @ freq_weights.T
                layer_noise = torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)
                noise_outputs.append(layer_noise)

            full_noise = torch.cat(noise_outputs, dim=-1)
            return full_noise


    # --- 2. NEURAL DETAIL MAPPER ---
    class NeuralDetailMapper(nn.Module):
        def __init__(self):
            super().__init__()
            self.grid = nn.Parameter(torch.randn(1, GRID_CHANNELS, GRID_SIZE, GRID_SIZE) * 0.1)
            self.noise_gen = MultiOctaveNoiseGenerator(NOISE_CHANNELS, OCTAVES, BASE_SCALE)

            input_dim = GRID_CHANNELS + NOISE_CHANNELS
            self.mlp = nn.Sequential(
                nn.Linear(input_dim, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(HIDDEN_DIM, 3),
                nn.Sigmoid()
            )

        def forward(self, coords):
            """
            Args:
                coords: (batch_size, 2) tensor with [x, y] coordinates in [0, 1]
            Returns:
                (batch_size, 3) RGB predictions
            """
            batch_size = coords.shape[0]

            # Get noise features
            v_noise = self.noise_gen(coords)

            # Sample grid features using bilinear interpolation
            # Normalize coords to [-1, 1] for grid_sample
            normalized_coords = coords * 2 - 1  # [0, 1] -> [-1, 1]
            grid_sample_input = normalized_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, batch_size, 2)

            # grid_sample expects (N, C, H, W) grid and (N, H_out, W_out, 2) grid_locations
            # Reshape for grid_sample
            grid_locations = normalized_coords.unsqueeze(0).unsqueeze(0)  # (1, 1, batch_size, 2)

            sampled_grid = torch.nn.functional.grid_sample(
                self.grid,
                grid_locations,
                mode='bilinear',
                align_corners=False,
                padding_mode='border'
            )  # (1, GRID_CHANNELS, 1, batch_size)

            v_grid = sampled_grid.squeeze(2).squeeze(0).T  # (batch_size, GRID_CHANNELS)

            # Concatenate features and pass through MLP
            features = torch.cat([v_grid, v_noise], dim=-1)
            output = self.mlp(features)

            return output


    dataset = TensorDataset(coords, torch.tensor(colors))
    loader = DataLoader(dataset, batch_size=1024, shuffle=True)
    model = NeuralDetailMapper().to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    is_cuda = (str(device) == 'cuda')
    print(is_cuda)
    print("Training starts")
    start = time.time()

    for epoch in range(50):
        epoch_start = time.time()
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')

    print("Training complete!")

    smp.reconstruct_image(model, device, H, W)