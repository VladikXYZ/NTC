import time

import matplotlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from scipy.ndimage import zoom
from torch import meshgrid
from torch.utils.data import TensorDataset, DataLoader

matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt


def image_init(img: np.ndarray):
    height, width = img.shape[:2]
    x_corners = np.stack([np.arange(width+1)-1, np.arange(width+1)], axis=0)  # left, right
    y_corners = np.stack([np.arange(height+1)-1, np.arange(height+1)], axis=0)  # top, bottom
    x_corners[0,0] = 0
    y_corners[0,0] = 0
    x_corners[1,-1] = width-1
    y_corners[1,-1] = height-1
    corners = np.stack([
        img[np.ix_(y_corners[0], x_corners[0])],  # top-left
        img[np.ix_(y_corners[0], x_corners[1])],  # top-right
        img[np.ix_(y_corners[1], x_corners[0])],  # bottom-left
        img[np.ix_(y_corners[1], x_corners[1])]  # bottom-right
    ], axis=0)
    samples = np.mean(corners, axis=0)  # (H+1,W+1,3)
    # plt.imshow(samples)
    # plt.show()
    y_coords, x_coords = np.meshgrid(np.linspace(0,1,height+1), np.linspace(0,1,width + 1))
    indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)  # (N, 2)
    return samples, indices

def upsample(img, scale):
    samples = zoom(img, (scale, scale, 1), order=1)  # order=1 = bilinear
    h, w = samples.shape[:2]
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)  # (N, 2)
    samples = samples.reshape(-1, 3)
    return indices, samples


img = Image.open("aa.png").convert('RGB')
img_array = np.array(img, dtype=np.float32) / 255.0
og, test_indices = image_init(img_array)
all_X, all_y = [], []
for scale in [0.5, 1, 2, 4, 6]:
    X, y = upsample(og, scale)
    all_X.append(X)
    all_y.append(y)
    # print(f"Scale {scale}: X={X.shape}, y={y.shape}")
X = np.vstack(all_X)  # (total_samples, input_features)
y = np.vstack(all_y)  # (total_samples, 3)

class ColorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 64), nn.ReLU(),
            nn.Linear(64, 3), nn.Sigmoid()
        )
    def forward(self, x):
        return torch.clamp(self.net(x), 0, 1)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device("cpu")
print(f"Using device: {device}")


model = ColorMLP().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)  # <- AFTER .to(device)

# ✅ Tensors to GPU
inputs_tensor = torch.from_numpy(X).float().to(device)
targets_tensor = torch.from_numpy(y).float().to(device)
dataset = TensorDataset(inputs_tensor, targets_tensor)
loader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=False)


from torch.amp import autocast, GradScaler
scaler = GradScaler('cuda')

print("Training started")
start = time.time()
for epoch in range(100):
    model.train()
    for batch_inputs, batch_targets in loader:
        optimizer.zero_grad()
        with autocast('cuda'):  # Mixed precision
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    print(f'Epoch {epoch}, Loss: {loss.item():.8f}')
print(f"Training took: {time.time() - start:.2f}s")

h, w = og.shape[:2]
with torch.no_grad():
    vertex_uvs = torch.from_numpy(test_indices).float().to(device)
    predicted_colors = model(vertex_uvs)
    predicted_colors = predicted_colors.cpu().numpy().reshape(h, w, 3)*255
    predicted_colors = predicted_colors.astype(dtype=np.uint8)
    og_np = og.cpu().numpy() if torch.is_tensor(og) else og

Image.fromarray(predicted_colors).save("reconstructed.png")
print("✓ Reconstructed image saved: reconstructed.png")

z = 16
h, w = z*h, z*w
y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
test_indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)  # (N, 2)
with torch.no_grad():
    vertex_uvs = torch.from_numpy(test_indices).float().to(device)
    predicted_colors = model(vertex_uvs)
    predicted_colors = predicted_colors.cpu().numpy().reshape(h, w, 3)*255
    predicted_colors = predicted_colors.astype(dtype=np.uint8)
    og_np = og.cpu().numpy() if torch.is_tensor(og) else og

Image.fromarray(predicted_colors).save(f"reconstructed{z}x.png")
print(f"✓ Reconstructed image with {z} zoom saved: reconstructed{z}x.png")
# fig, axes = plt.subplots(1, 3, figsize=(15, 5))
# axes[0].imshow(og_np); axes[0].set_title('Original'); axes[0].axis('off')
# axes[1].imshow(predicted_colors); axes[1].set_title('MLP Output'); axes[1].axis('off')
# diff = np.mean(np.abs(predicted_colors - og_np), axis=2)
# im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
# axes[2].set_title('Difference'); axes[2].axis('off')
# plt.tight_layout()
# plt.show()
