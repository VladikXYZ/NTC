import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
import preprocessing as prep


class ColorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.ReLU(),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return torch.clamp(self.net(x), 0, 1)




X,y, test_indices, og = prep.pipeline("aa.png", [6])
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
    predicted_colors = predicted_colors.cpu().numpy().reshape(h, w, 3)
    og_np = og.cpu().numpy() if torch.is_tensor(og) else og

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(og_np); axes[0].set_title('Original'); axes[0].axis('off')
axes[1].imshow(predicted_colors); axes[1].set_title('MLP Output'); axes[1].axis('off')
diff = np.mean(np.abs(predicted_colors - og_np), axis=2)
im = axes[2].imshow(diff, cmap='hot', vmin=0, vmax=0.1)
axes[2].set_title('Difference'); axes[2].axis('off')
plt.tight_layout()
plt.show()