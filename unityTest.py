import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import os

# --- SETTINGS ---
GRID_SIZE = 512
HIDDEN_DIM = 64
INPUT_DIM = 40
OCTAVES = 4
DATA_DIR = "Output/UnityData"

device = torch.device("cpu")

def load_tensor(filename, shape):
    path = os.path.join(DATA_DIR, filename)
    data = np.fromfile(path, dtype=np.float32)
    t = torch.from_numpy(data.astype(np.float32))
    if shape:
        return t.reshape(shape)
    return t

def run_reconstruction():
    print("Reconstructing image from .bytes files...")

    # 1. Load Data
    # saved grid was grid_full[..., 0:4]/[...,4:8] i.e. H,W,channels
    grid_a = load_tensor("grid_texture_a.bytes", (GRID_SIZE, GRID_SIZE, 4)).permute(2,0,1).unsqueeze(0)  # (1,4,Hg,Wg)
    grid_b = load_tensor("grid_texture_b.bytes", (GRID_SIZE, GRID_SIZE, 4)).permute(2,0,1).unsqueeze(0)
    noise_freqs = load_tensor("noise_freqs.bytes", None).numpy().astype(np.float32)  # use numpy for indexing
    l0_w = load_tensor("l0_weight.bytes", (HIDDEN_DIM, INPUT_DIM))
    l0_b = load_tensor("l0_bias.bytes", (HIDDEN_DIM,))
    l1_w = load_tensor("l1_weight.bytes", (HIDDEN_DIM, HIDDEN_DIM))
    l1_b = load_tensor("l1_bias.bytes", (HIDDEN_DIM,))
    l2_w = load_tensor("l2_weight.bytes", (3, HIDDEN_DIM))
    l2_b = load_tensor("l2_bias.bytes", (3,))

    # 2. Output resolution
    H, W = 512, 512
    y = torch.linspace(0.0, 1.0, H, dtype=torch.float32)
    x = torch.linspace(0.0, 1.0, W, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing='ij')  # yy: rows, xx: cols
    flat_x = xx.flatten()  # length H*W
    flat_y = yy.flatten()

    # 3. Grid Sampling (coords in -1..1, order: x,y)
    sample_x = (flat_x * 2.0 - 1.0).reshape(1, H, W)
    sample_y = (flat_y * 2.0 - 1.0).reshape(1, H, W)
    # stack to (1,H,W,2) and put x then y (PyTorch grid uses x,y)
    grid_coords = torch.stack([sample_x, sample_y], dim=-1).squeeze(0)  # (H,W,2)
    grid_coords = grid_coords.unsqueeze(0)  # (1,H,W,2)

    # Use same align_corners as training
    out_a = F.grid_sample(grid_a, grid_coords, align_corners=False, mode='bilinear')  # (1,4,H,W)
    out_b = F.grid_sample(grid_b, grid_coords, align_corners=False, mode='bilinear')

    # Remove batch and flatten to (num_pixels, channels)
    num_pixels = H * W
    feat_a = out_a.squeeze(0).reshape(4, num_pixels).T  # (num_pixels, 4)
    feat_b = out_b.squeeze(0).reshape(4, num_pixels).T  # (num_pixels, 4)

    # 4. Noise — reconstruct per-octave exactly like training:
    # training stored freq layers in order: for each octave freq_weights (4,2) flattened -> concat octaves
    ptr = 0
    noise_channels_list = []
    for octave in range(OCTAVES):
        # each octave has 4 frequency vectors (wx,wy)
        # gather all 4 projections first (to match model's projection shape (h,w,4))
        projs = []
        for k in range(4):
            wx = float(noise_freqs[ptr]); wy = float(noise_freqs[ptr+1]); ptr += 2
            proj = flat_x * wx + flat_y * wy  # shape (num_pixels,)
            projs.append(proj)
        # now projs is [proj0, proj1, proj2, proj3]
        proj_stack = torch.stack(projs, dim=1)  # (num_pixels, 4)
        # append all sin channels for this octave, then all cos channels (same order as model)
        noise_channels_list.append(torch.sin(proj_stack))   # (num_pixels,4)
        noise_channels_list.append(torch.cos(proj_stack))   # (num_pixels,4)

    # concat across channels -> (num_pixels, noise_channels=32)
    noise_tensor = torch.cat(noise_channels_list, dim=1).to(torch.float32)

    # 5. Build input and run MLP (as linear layers)
    input_feats = torch.cat([feat_a, feat_b, noise_tensor], dim=1)  # (num_pixels, 40)
    h0 = F.relu(F.linear(input_feats, l0_w, l0_b))
    h1 = F.relu(F.linear(h0, l1_w, l1_b))
    rgb = torch.sigmoid(F.linear(h1, l2_w, l2_b))  # (num_pixels,3)

    # 6. Save
    rgb_img = (rgb.reshape(H, W, 3).numpy() * 255.0).astype(np.uint8)
    img = Image.fromarray(rgb_img)
    os.makedirs("Output", exist_ok=True)
    img.save("Output/PROOF_OF_DATA.png")
    print("Saved Output/PROOF_OF_DATA.png")

if __name__ == "__main__":
    run_reconstruction()
