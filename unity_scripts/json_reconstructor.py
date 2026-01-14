import json
import numpy as np
from PIL import Image
import os
import argparse
import math

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def relu(x):
    return np.maximum(0, x)

class SimulatedShader:
    def __init__(self, json_path):
        print(f"Loading model from {json_path}...")
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.layers = {}
        for layer in data['layers']:
            self.layers[layer['name']] = {
                'shape': layer['shape'],
                'data': np.array(layer['data'], dtype=np.float32)
            }
            
        # 1. Parse Metadata or Infer
        self.levels = 6
        self.feature_dim = 4
        
        if 'metadata' in data:
            self.levels = data['metadata'].get('levels', 6)
            self.feature_dim = data['metadata'].get('feature_dim', 4)
            print(f"Metadata found: Levels={self.levels}, Feats={self.feature_dim}")
        
        # 2. Extract Grids (grids.0, grids.1, ...)
        self.grids = {}
        for i in range(self.levels):
            key = f"grids.{i}"
            if key in self.layers:
                layer = self.layers[key]
                # Shape: [1, FeatureDim, H, W]
                # PyTorch shape is (N, C, H, W)
                c = layer['shape'][1]
                h = layer['shape'][2]
                w = layer['shape'][3]
                
                # Reshape flat data to [C, H, W]
                # Note: PyTorch flatten order is Row-Major (C-style)
                # data[c][h][w]
                grid_data = layer['data'].reshape(c, h, w)
                self.grids[i] = grid_data
                print(f"Loaded Grid {i}: {w}x{h}x{c}")
            else:
                print(f"Warning: Grid {i} missing from JSON!")

        # 3. Extract MLP Weights
        # Input dim = levels * feature_dim
        input_dim = self.levels * self.feature_dim
        
        self.w1 = self.get_weights("mlp.0.weight", (64, input_dim))
        self.b1 = self.get_weights("mlp.0.bias", (64,))
        self.w2 = self.get_weights("mlp.2.weight", (64, 64))
        self.b2 = self.get_weights("mlp.2.bias", (64,))
        self.w3 = self.get_weights("mlp.4.weight", (3, 64))
        self.b3 = self.get_weights("mlp.4.bias", (3,))
        
    def get_weights(self, name, shape):
        if name not in self.layers:
            raise ValueError(f"Missing layer: {name}")
        return self.layers[name]['data'].reshape(shape)

    def bilinear_sample_grid(self, grid_data, u, v):
        # grid_data shape: [C, H, W]
        c, h, w = grid_data.shape
        
        # Map u,v [0, 1] to pixel coords [0, W]
        # Align corners = True logic in PyTorch grid_sample usually means:
        # (-1, -1) -> (0, 0), (1, 1) -> (W-1, H-1)
        # Here we have 0..1 input.
        # "align_corners=True" generally maps 0.0 to index 0 and 1.0 to index N-1
        
        x = u * (w - 1)
        y = v * (h - 1)
        
        x0 = int(np.floor(x))
        y0 = int(np.floor(y))
        x1 = x0 + 1
        y1 = y0 + 1
        
        dx = x - x0
        dy = y - y0
        
        # Clamp
        x0 = np.clip(x0, 0, w - 1)
        x1 = np.clip(x1, 0, w - 1)
        y0 = np.clip(y0, 0, h - 1)
        y1 = np.clip(y1, 0, h - 1)
        
        # Fetch corners [C]
        c00 = grid_data[:, y0, x0]
        c10 = grid_data[:, y0, x1]
        c01 = grid_data[:, y1, x0]
        c11 = grid_data[:, y1, x1]
        
        # Interpolate
        c0 = c00 * (1 - dx) + c10 * dx
        c1 = c01 * (1 - dx) + c11 * dx
        val = c0 * (1 - dy) + c1 * dy
        
        return val

    def infer_pixel(self, u, v):
        # 1. Feature Extraction (Pyramid)
        features = []
        for i in range(self.levels):
            if i in self.grids:
                # Sample from grid i
                feat = self.bilinear_sample_grid(self.grids[i], u, v)
                features.append(feat)
            else:
                features.append(np.zeros(self.feature_dim))
        
        # Concatenate all features
        # [f0, f1, f2...]
        input_vec = np.concatenate(features)

        # 2. MLP Forward
        # Layer 1: Input -> 64
        # w1 is [64, Input]
        h1 = self.w1 @ input_vec + self.b1
        h1 = relu(h1)
        
        # Layer 2: 64 -> 64
        h2 = self.w2 @ h1 + self.b2
        h2 = relu(h2)
        
        # Layer 3: 64 -> 3
        out = self.w3 @ h2 + self.b3
        rgb = sigmoid(out)
        
        return rgb

    def render(self, width, height, output_name="reconstructed_json.png"):
        print(f"Rendering {width}x{height} image...")
        image_data = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Optimization: Pre-compute coordinates grid if needed, 
        # but pure python loop is slow anyway, good for verification.
        
        for y in range(height):
            if y % 20 == 0: print(f"Row {y}/{height}", end='\r')
            v = (y + 0.5) / height # Pixel center
            for x in range(width):
                u = (x + 0.5) / width # Pixel center
                
                rgb = self.infer_pixel(u, v)
                
                image_data[y, x] = (rgb * 255).astype(np.uint8)
                
        print("\nUsing PIL to save...")
        img = Image.fromarray(image_data)
        img.save(output_name)
        print(f"✓ Saved to {output_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file", help="Path to the JSON weights file")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--out", default="reconstructed_json.png")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.json_file):
        print(f"Error: File {args.json_file} does not exist.")
    else:
        sim = SimulatedShader(args.json_file)
        sim.render(args.width, args.height, args.out)
