# train_and_export_unity.py
import time
import json
import os
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import sampling as smp  # your existing sampling helper

# ---------------------------
# CONFIG
# ---------------------------
OUTPUT_DIR = "Output"
UNITY_DIR = os.path.join(OUTPUT_DIR, "UnityData")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(UNITY_DIR, exist_ok=True)

BATCH_CPU = 256
EPOCHS = 10
LR = 0.01
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ---------------------------
# UTIL: packing helpers
# ---------------------------
def pad_to_multiple(arr, multiple):
    n = arr.shape[-1]
    pad = (multiple - (n % multiple)) % multiple
    if pad == 0:
        return arr
    pad_shape = list(arr.shape[:-1]) + [pad]
    pad_arr = np.zeros(pad_shape, dtype=arr.dtype)
    return np.concatenate([arr, pad_arr], axis=-1)

def pack_to_float4(arr2d):
    """
    arr2d: numpy array shape (rows, cols) of dtype float32
    Pack last dimension into groups of 4 floats -> returns np.float32 array of shape (rows * (cols/4), 4)
    That layout matches StructuredBuffer<float4> indexed by [row * chunks + chunkIdx]
    """
    rows, cols = arr2d.shape
    assert cols % 4 == 0, "cols must be divisible by 4 (pad first)"
    chunks = cols // 4
    out = np.zeros((rows * chunks, 4), dtype=np.float32)
    for r in range(rows):
        for c in range(chunks):
            base = c * 4
            out[r * chunks + c, :] = arr2d[r, base:base + 4]
    return out

def save_f32(path, arr):
    """Save numpy float32 array as contiguous float32 bytes."""
    arr.astype(np.float32).tofile(path)
    print(f"Saved {path} | shape={arr.shape} | dtype={arr.dtype}")

# ---------------------------
# MODEL definition (same)
# ---------------------------
class ColorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 128), nn.Sigmoid(),
            nn.Linear(128, 64), nn.Sigmoid(),
            nn.Linear(64, 3), nn.Sigmoid())

    def forward(self, x):
        return self.net(x)

# ---------------------------
# MAIN: training + export
# ---------------------------
def main():
    print("Device:", DEVICE)
    img = Image.open("Input/oa.png").convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    og, test_indices = smp.image_init(img_array)

    all_X, all_y = [], []
    for scale in [0.25, 0.5, 1]:
        X, y = smp.upsample(og, scale)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.vstack(all_y)

    inputs_tensor = torch.from_numpy(X).float()
    targets_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(inputs_tensor, targets_tensor)
    is_cuda = (str(DEVICE) == 'cuda')
    batch_size = max(BATCH_CPU, 4096 * (4 if is_cuda else 1))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=12 if is_cuda else 0, pin_memory=is_cuda)

    model = ColorMLP().to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("Training starts")
    for epoch in range(EPOCHS):
        epoch_start = time.time()
        for batch_inputs, batch_targets in loader:
            batch_inputs = batch_inputs.to(DEVICE, non_blocking=True)
            batch_targets = batch_targets.to(DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')

    # save a debug reconstruction image (uses your reconstruct helper)
    h, w = og.shape[:2]
    smp.reconstruct_image(model, DEVICE, h, w)
    smp.reconstruct_image(model, DEVICE, h, w, 2, bench=True)

    # ---------------------------
    # EXPORT PARAMETERS to UnityData
    # ---------------------------
    # extract weights and biases
    l0 = model.net[0]  # Linear(2 -> 128)
    l1 = model.net[2]  # Linear(128 -> 64)
    l2 = model.net[4]  # Linear(64 -> 3)

    # convert to numpy (row-major: out_features x in_features)
    l0_w = l0.weight.detach().cpu().numpy().astype(np.float32)  # shape (128, 2)
    l0_b = l0.bias.detach().cpu().numpy().astype(np.float32)    # shape (128,)

    l1_w = l1.weight.detach().cpu().numpy().astype(np.float32)  # shape (64, 128)
    l1_b = l1.bias.detach().cpu().numpy().astype(np.float32)    # shape (64,)

    l2_w = l2.weight.detach().cpu().numpy().astype(np.float32)  # shape (3, 64)
    l2_b = l2.bias.detach().cpu().numpy().astype(np.float32)    # shape (3,)

    # Pad each weight row width to multiple of 4 and pack into float4 arrays
    # L0: (128, 2) -> pad to 4 -> (128,4) -> chunks_input = 1
    l0_w_padded = pad_to_multiple(l0_w, 4)  # -> (128,4)
    l0_w4 = pack_to_float4(l0_w_padded)     # -> (128*1, 4)
    # L1: (64,128) -> already 128 multiple of 4 -> chunks = 32
    l1_w_padded = pad_to_multiple(l1_w, 4)  # (64,128)
    l1_w4 = pack_to_float4(l1_w_padded)     # -> (64*32, 4)
    # L2: (3,64) -> 64 is multiple of 4 -> chunks = 16
    l2_w_padded = pad_to_multiple(l2_w, 4)  # (3,64)
    l2_w4 = pack_to_float4(l2_w_padded)     # -> (3*16, 4)

    # save raw biases as f32
    save_f32(os.path.join(UNITY_DIR, "l0_bias.bytes"), l0_b)
    save_f32(os.path.join(UNITY_DIR, "l1_bias.bytes"), l1_b)
    save_f32(os.path.join(UNITY_DIR, "l2_bias.bytes"), l2_b)

    # save packed weight float4 arrays
    # stored as contiguous float32 values in the order expected by shader: neuron0_chunk0, neuron0_chunk1, ...
    save_f32(os.path.join(UNITY_DIR, "l0_weight4.bytes"), l0_w4.flatten())
    save_f32(os.path.join(UNITY_DIR, "l1_weight4.bytes"), l1_w4.flatten())
    save_f32(os.path.join(UNITY_DIR, "l2_weight4.bytes"), l2_w4.flatten())

    # also save raw float backups if needed
    save_f32(os.path.join(UNITY_DIR, "l0_weight_raw.bytes"), l0_w.flatten())
    save_f32(os.path.join(UNITY_DIR, "l1_weight_raw.bytes"), l1_w.flatten())
    save_f32(os.path.join(UNITY_DIR, "l2_weight_raw.bytes"), l2_w.flatten())

    # Write metadata for Unity shader / loader
    metadata = {
        "input_dim": 2,
        "h0": 128,
        "h1": 64,
        "out_dim": 3,
        "l0_chunks": l0_w_padded.shape[1] // 4,   # 1
        "l1_chunks": l1_w_padded.shape[1] // 4,   # 32
        "l2_chunks": l2_w_padded.shape[1] // 4,   # 16
        "note": "Weights are stored packed as float4 arrays (rows * chunks, 4). Bind them as StructuredBuffer<float4> from Unity."
    }
    with open(os.path.join(UNITY_DIR, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)
    print("Exported UnityData to", UNITY_DIR)
    print("Metadata:", metadata)

if __name__ == "__main__":
    main()
