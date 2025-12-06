import time
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using: {device}")

    # Load image
    img = Image.open("oa.png").convert("RGB")
    target = ToTensor()(img).unsqueeze(0).to(device)
    _, _, H, W = target.shape
    print(f"Training resolution: {W}x{H}")

    # Generate coordinate grid dataset
    y = torch.linspace(0, 1, H, device=device)
    x = torch.linspace(0, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    coords = torch.stack([xx, yy], dim=-1).reshape(-1, 2)  # Flatten to [H*W, 2]
    colors = target.squeeze(0).permute(1, 2, 0).reshape(-1, 3)  # Flatten to [H*W, 3]


    class NeuralTexture(nn.Module):
        def __init__(self):
            super().__init__()
            # Learnable feature grid (low-freq details)
            self.feature_map = nn.Parameter(torch.randn(1, 32, 128, 128) * 0.1)
            # Fixed noise frequencies (high-freq details)
            self.freqs = torch.randn(16, 2, device=device) * 20.0
            # Small decoder: features -> RGB
            self.decoder = nn.Sequential(
                nn.Linear(32 + 32, 64), nn.ReLU(),
                nn.Linear(64, 3), nn.Sigmoid()
            )

        def forward(self, coord_batch):
            batch_size = coord_batch.shape[0]

            # Sample from feature grid via bilinear interpolation
            # Reshape coords for grid_sample: [batch, H, W, 2]
            sample_coords = coord_batch.unsqueeze(0).unsqueeze(0) * 2 - 1  # [1, 1, batch, 2]
            sample_coords = sample_coords.flip(-1)  # x,y order for grid_sample

            # Expand feature map to match batch size
            feature_map_batch = self.feature_map.expand(batch_size, -1, -1, -1)

            grid_feat = torch.nn.functional.grid_sample(
                feature_map_batch, sample_coords, align_corners=False
            )  # [batch, 32, 1, 1]
            grid_feat = grid_feat.squeeze(-1).squeeze(-1)  # [batch, 32]

            # Generate noise features via random frequency projection
            proj = coord_batch @ self.freqs.T
            noise_feat = torch.cat([torch.sin(proj), torch.cos(proj)], dim=-1)

            # Combine and decode to RGB
            combined = torch.cat([grid_feat, noise_feat], dim=-1)
            return self.decoder(combined)


    # Dataset & DataLoader
    dataset = TensorDataset(coords, colors)
    loader = DataLoader(dataset, batch_size=256, shuffle=True)

    model = NeuralTexture().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    scaler = torch.GradScaler('cuda') if device.type == 'cuda' else None

    print("Training starts")
    start = time.time()

    for epoch in range(100):
        epoch_start = time.time()
        for batch_coords, batch_colors in loader:
            batch_coords = batch_coords.to(device)
            batch_colors = batch_colors.to(device)

            optimizer.zero_grad()

            if scaler:
                with torch.autocast(device.type):
                    preds = model(batch_coords)
                    loss = criterion(preds, batch_colors)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                preds = model(batch_coords)
                loss = criterion(preds, batch_colors)
                loss.backward()
                optimizer.step()

        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')

    # Render full resolution
    print("\nRendering...")
    with torch.no_grad():
        output = model(coords).reshape(H, W, 3).permute(2, 0, 1)
    ToPILImage()(output.cpu()).save("output.png")
    print("Saved output.png")

    print(f"Total training time: {time.time() - start:.2f}s")
