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

INPUT_IMAGE = "lama.jpg"

# --- KONFIGURACE ---
CONFIG = {
    'grid_size': 128,
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

# --- 1. GENERÁTOR ŠUMU ---
class MultiOctaveNoiseGenerator(nn.Module):
    def __init__(self, total_channels, octaves, base_scale):
        super().__init__()
        self.octaves = octaves
        self.channels_per_octave = total_channels // octaves
        self.freq_layers = nn.ParameterList([])
        
        for i in range(octaves):
            current_scale = base_scale * (2 ** i)
            w_shape = (self.channels_per_octave // 2, 2)
            # Fixní náhodné frekvence - uloží se do state_dict
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

# --- 2. UPRAVENÝ MODEL (bere config v __init__) ---
class NeuralDetailMapper(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config # Uložíme si config pro jistotu
        
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
        # Větev B: Šum
        v_noise = self.noise_gen(h, w, y_range, x_range)
        
        # Větev A: Grid
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
        # Training pass - globální H, W
        features = self.get_features(H, W, (0,1), (0,1))
        return self.mlp(features)

    def render_tile(self, out_h, out_w, y_range, x_range):
        features = self.get_features(out_h, out_w, y_range, x_range)
        return self.mlp(features)

# Inicializace s configem
model = NeuralDetailMapper(CONFIG).to(device)
optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
loss_fn = nn.MSELoss()
scaler = torch.cuda.amp.GradScaler()

# --- TRAIN ---
print(f"Training with {CONFIG['octaves']} octaves...")
patience = 200 
best_loss = float("inf")
no_improve = 0

pbar = tqdm(total=patience, desc="Learning Texture")

while True:
    optimizer.zero_grad()
    with torch.cuda.amp.autocast():
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

    if no_improve >= patience:
        break

pbar.close()

# --- 3. UKLÁDÁNÍ (SAVE) ---
print("\nSaving model checkpoint...")
checkpoint_path = "lama_model.pth"

torch.save({
    'model_state_dict': model.state_dict(), # Toto uloží GRID (feature mapy), váhy MLP i frekvence šumu
    'config': CONFIG,                       # Toto uloží nastavení (128 grid size, 4 octaves atd.)
    'training_resolution': (H, W)           # Hodí se vědět, na jakém rozlišení se trénovalo
}, checkpoint_path)

print(f"Model saved to {checkpoint_path}")

# --- 4. EXPERIMENT: JAK TO PAK NAČÍST (LOAD) ---
# Tento kód můžeš použít v úplně novém souboru, kde nebudeš trénovat.

def load_trained_model(path):
    print(f"Loading from {path}...")
    checkpoint = torch.load(path, map_location=device)
    
    # 1. Načteme config
    loaded_config = checkpoint['config']
    
    # 2. Vytvoříme PRÁZDNY model podle configu
    new_model = NeuralDetailMapper(loaded_config).to(device)
    
    # 3. Nasypeme tam naučené váhy
    new_model.load_state_dict(checkpoint['model_state_dict'])
    new_model.eval() # Přepnout do eval módu (vypne dropou, fixne batchnorm kdyby tam byl)
    
    print("Model loaded successfully!")
    return new_model

# Test exportu s "nově" načteným modelem
loaded_model = load_trained_model(checkpoint_path)

with torch.no_grad():
    # Render native resolution
    print("Rendering native resolution from loaded model...")
    recon = loaded_model.render_tile(H, W, (0,1), (0,1))[0].detach().cpu()
    ToPILImage()(recon).save("output_loaded_native.png")
    print("Saved output_loaded_native.png")

print("Done.")