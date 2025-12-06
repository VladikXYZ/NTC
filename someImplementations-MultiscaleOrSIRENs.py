import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from PIL import Image

class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True, is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.linear.in_features, 
                                             1 / self.linear.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.linear.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.linear.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

class MlpNetwork(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=64, layers=3):
        super().__init__()
        net = []
        net.append(SineLayer(in_dim, hidden_dim, is_first=True))
        for _ in range(layers - 2):
            net.append(SineLayer(hidden_dim, hidden_dim))
        self.linear_out = nn.Linear(hidden_dim, out_dim)

        with torch.no_grad():
            self.linear_out.weight.uniform_(-np.sqrt(6 / hidden_dim) / 30, np.sqrt(6 / hidden_dim) / 30)

        net.append(self.linear_out)
        self.net = nn.Sequential(*net)

    def forward(self, x):
        return self.net(x)


# ============================================================================
# OPTION 1: Multi-Scale Grid (Recommended - Best Balance)
# ============================================================================
class MultiScaleTextureNetwork(nn.Module):
    """
    Samples a single high-res grid at multiple frequencies and combines them.
    Similar to Perlin noise but learned.
    """
    def __init__(self, aspect_ratio, base_res=1024, feature_dim=32, num_octaves=4):
        super().__init__()

        if aspect_ratio > 1:
            self.h_grid = int(base_res * aspect_ratio)
            self.w_grid = base_res
        else:
            self.h_grid = base_res
            self.w_grid = int(base_res / aspect_ratio)

        print(f"Multi-Scale Grid: {self.w_grid}x{self.h_grid}, {feature_dim} channels, {num_octaves} octaves")

        self.feature_dim = feature_dim
        self.num_octaves = num_octaves
        
        # Single high-resolution feature grid
        self.grid = nn.Parameter(torch.randn(1, feature_dim, self.h_grid, self.w_grid) * 0.1)
        
        # Learnable weights for each octave
        self.octave_weights = nn.Parameter(torch.ones(num_octaves) / num_octaves)
        
        # Color decoder
        self.color_net = MlpNetwork(feature_dim, 3, hidden_dim=128, layers=4)

    def sample_grid(self, coords):
        """Sample from grid using bilinear interpolation"""
        coords = coords % 1.0
        grid_coords = coords.view(1, -1, 1, 2).to(self.grid.device) * 2.0 - 1.0
        
        features = F.grid_sample(
            self.grid,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        
        return features.squeeze(-1).squeeze(0).permute(1, 0)

    def forward(self, coords):
        batch_size = coords.shape[0]
        accumulated = torch.zeros(batch_size, self.feature_dim, device=coords.device)
        
        # Sample at different frequencies (classic multi-scale approach)
        weights = F.softmax(self.octave_weights, dim=0)
        
        for i in range(self.num_octaves):
            scale = 2.0 ** i  # 1, 2, 4, 8, ...
            octave_features = self.sample_grid(coords * scale)
            accumulated += octave_features * weights[i]
        
        rgb = self.color_net(accumulated)
        return torch.sigmoid(rgb)


# ============================================================================
# OPTION 2: Simple High-Res Grid (Simplest - Good Baseline)
# ============================================================================
class SimpleGridTextureNetwork(nn.Module):
    """
    Just a high-resolution feature grid + decoder network.
    No multi-scale, just direct mapping.
    """
    def __init__(self, aspect_ratio, base_res=1024, feature_dim=32):
        super().__init__()

        if aspect_ratio > 1:
            self.h_grid = int(base_res * aspect_ratio)
            self.w_grid = base_res
        else:
            self.h_grid = base_res
            self.w_grid = int(base_res / aspect_ratio)

        print(f"Simple Grid: {self.w_grid}x{self.h_grid}, {feature_dim} channels")

        # Single high-resolution feature grid
        self.grid = nn.Parameter(torch.randn(1, feature_dim, self.h_grid, self.w_grid) * 0.1)
        
        # Color decoder
        self.color_net = MlpNetwork(feature_dim, 3, hidden_dim=64, layers=3)

    def sample_grid(self, coords):
        """Sample from grid using bilinear interpolation"""
        coords = coords % 1.0
        grid_coords = coords.view(1, -1, 1, 2).to(self.grid.device) * 2.0 - 1.0
        
        features = F.grid_sample(
            self.grid,
            grid_coords,
            mode="bilinear",
            padding_mode="border",
            align_corners=False
        )
        
        return features.squeeze(-1).squeeze(0).permute(1, 0)

    def forward(self, coords):
        features = self.sample_grid(coords)
        rgb = self.color_net(features)
        return torch.sigmoid(rgb)


# ============================================================================
# OPTION 3: Pure SIREN (No Grid - Continuous Function)
# ============================================================================
class SIRENTextureNetwork(nn.Module):
    """
    Pure coordinate-based network. No grid at all.
    True infinite resolution but slower to train.
    """
    def __init__(self, hidden_dim=256, layers=5):
        super().__init__()
        
        print(f"SIREN Network: {layers} layers, {hidden_dim} hidden dim")
        
        # Build SIREN network: coords (2D) -> RGB (3D)
        net = []
        net.append(SineLayer(2, hidden_dim, is_first=True, omega_0=30))
        
        for _ in range(layers - 2):
            net.append(SineLayer(hidden_dim, hidden_dim, omega_0=30))
        
        # Final layer
        self.final_linear = nn.Linear(hidden_dim, 3)
        with torch.no_grad():
            self.final_linear.weight.uniform_(-np.sqrt(6 / hidden_dim) / 30, 
                                              np.sqrt(6 / hidden_dim) / 30)
        
        net.append(self.final_linear)
        self.net = nn.Sequential(*net)

    def forward(self, coords):
        # SIREN works best with coords in [-1, 1] range
        coords_scaled = coords * 2.0 - 1.0
        rgb = self.net(coords_scaled)
        return torch.sigmoid(rgb)


# ============================================================================
# Training Function
# ============================================================================
def train_one_image(target_img_path, model_type="multiscale", steps=5000):
    """
    model_type: "multiscale", "simple", or "siren"
    """
    from torchvision.transforms import ToTensor

    if not os.path.exists(target_img_path):
        print("Error: File not found")
        return None, 0, 0

    print(f"Loading {target_img_path}...")
    pil_img = Image.open(target_img_path).convert('RGB')
    W, H = pil_img.size
    print(f"Resolution: {W}x{H}")

    target_tensor = ToTensor()(pil_img).permute(1, 2, 0).cuda()
    aspect = H / W
    
    # Choose model architecture
    if model_type == "multiscale":
        grid_res = min(max(W, H), 2048)  # Match texture size, cap at 2048
        model = MultiScaleTextureNetwork(
            aspect_ratio=aspect,
            base_res=grid_res,
            feature_dim=32,
            num_octaves=4
        ).cuda()
        lr_grid = 1e-2
        lr_net = 1e-3
        
    elif model_type == "simple":
        grid_res = min(max(W, H), 2048)
        model = SimpleGridTextureNetwork(
            aspect_ratio=aspect,
            base_res=grid_res,
            feature_dim=32
        ).cuda()
        lr_grid = 1e-2
        lr_net = 5e-4
        
    elif model_type == "siren":
        model = SIRENTextureNetwork(
            hidden_dim=256,
            layers=5
        ).cuda()
        lr_grid = None  # No grid
        lr_net = 1e-4
        
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    # Setup optimizer
    if lr_grid is not None:
        optimizer = torch.optim.Adam([
            {'params': [model.grid], 'lr': lr_grid},
            {'params': [p for n, p in model.named_parameters() if 'grid' not in n], 'lr': lr_net}
        ])
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr_net)

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=steps, eta_min=1e-6)

    print(f"Training {model_type} model for {steps} steps...")

    # Pre-compute coordinates
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, H), 
        torch.linspace(0, 1, W), 
        indexing='ij'
    )
    all_coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)
    target_flat = target_tensor.reshape(-1, 3)

    batch_size = 2**14  # 16384
    num_pixels = all_coords.shape[0]
    
    best_loss = float('inf')

    for step in range(steps):
        optimizer.zero_grad()

        # Random batch
        idx = torch.randint(0, num_pixels, (batch_size,))
        batch_coords = all_coords[idx].cuda()
        batch_target = target_flat[idx].cuda()

        output = model(batch_coords)
        loss = F.mse_loss(output, batch_target)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        if loss.item() < best_loss:
            best_loss = loss.item()

        if step % 100 == 0:
            print(f"Step {step}/{steps}, Loss: {loss.item():.6f}, Best: {best_loss:.6f}, LR: {scheduler.get_last_lr()[0]:.2e}")

        # Validation check
        if step % 1000 == 0 and step > 0:
            model.eval()
            with torch.no_grad():
                test_idx = torch.randint(0, num_pixels, (min(20000, num_pixels),))
                test_out = model(all_coords[test_idx].cuda())
                test_loss = F.mse_loss(test_out, target_flat[test_idx].cuda())
                print(f"  → Validation loss: {test_loss.item():.6f}")
            model.train()

    return model, H, W


def save_result(model, h, w, filename):
    print(f"Rendering output to {filename}...")
    model.eval()

    grid_y, grid_x = torch.meshgrid(
        torch.linspace(0, 1, h),
        torch.linspace(0, 1, w),
        indexing='ij'
    )
    coords = torch.stack([grid_x, grid_y], dim=-1).reshape(-1, 2)

    full_img = torch.zeros((h*w, 3))
    batch_size = 50000
    
    with torch.no_grad():
        for i in range(0, coords.shape[0], batch_size):
            batch = coords[i:i+batch_size].cuda()
            out = model(batch)
            full_img[i:i+batch_size] = out.cpu()

            if i % (batch_size * 5) == 0:
                print(f"{i/coords.shape[0]*100:.1f}%")

    arr = (full_img.reshape(h, w, 3).numpy() * 255).astype(np.uint8)
    Image.fromarray(arr).save(filename)
    print(f"Saved to {filename}")


def export_feature_maps(model, output_folder="feature_maps"):
    """Export learned feature grid (only for grid-based models)"""
    import os
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    if not hasattr(model, 'grid'):
        print("Model has no grid to export (SIREN model)")
        torch.save(model.state_dict(), f"{output_folder}/model_weights.pth")
        return

    grid_data = model.grid.data.cpu().squeeze(0)
    channels, h, w = grid_data.shape
    print(f"Exporting feature grid: {w}x{h} with {channels} channels...")

    # Normalize to [0, 1]
    min_val = grid_data.min()
    max_val = grid_data.max()
    grid_norm = (grid_data - min_val) / (max_val - min_val + 1e-8)

    # Save as RGB images (3 channels at a time)
    for i in range(0, channels, 3):
        chunk = grid_norm[i : i+3]

        if chunk.shape[0] < 3:
            padding = torch.zeros((3 - chunk.shape[0], h, w))
            chunk = torch.cat([chunk, padding], dim=0)

        chunk_img = chunk.permute(1, 2, 0).numpy()
        filename = f"{output_folder}/features_ch{i:02d}-{i+2:02d}.png"
        img_uint8 = (chunk_img * 255).astype(np.uint8)
        Image.fromarray(img_uint8).save(filename)

    torch.save(model.state_dict(), f"{output_folder}/model_weights.pth")
    print(f"Saved to {output_folder}/")


if __name__ == "__main__":
    if torch.cuda.is_available():
        input_file = "Input/texture2.jpg"
        
        # Choose architecture:
        # "multiscale" - Best for most textures (recommended)
        # "simple"     - Fastest, good baseline
        # "siren"      - Pure neural, slowest but infinite resolution
        
        model_type = "multiscale" 
        
        print(f"\n{'='*60}")
        print(f"Training with {model_type.upper()} architecture")
        print(f"{'='*60}\n")
        
        model, h, w = train_one_image(input_file, model_type=model_type, steps=3000)

        if model:
            save_result(model, h, w, f"Output/texture2_{model_type}_final.png")
            export_feature_maps(model, output_folder=f"Output/texture2_{model_type}_data")
    else:
        print("CUDA required")