import os

import scipy as scp
import numpy as np
import torch
from PIL import Image
from torch.profiler import profile, record_function, ProfilerActivity

def load_image(path: os.PathLike): return np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def image_init(path: os.PathLike):
    img = load_image(path)
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

def upsample(img: np.ndarray[np.ndarray[np.float32]], scale):
    samples = scp.ndimage.zoom(img, (scale, scale, 1), order=1)  # order=1 = bilinear
    h, w = samples.shape[:2]
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)  # (N, 2)
    samples = samples.reshape(-1, 3)
    return indices, samples

def reconstruct_image(model, device, height, width, z = 1, bench = False, name = "reconstructed"):
    model.eval()
    h, w = z*height, z*width
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    test_indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
    with torch.no_grad():
        vertex_uvs = torch.from_numpy(test_indices).float().to(device)
        if bench:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, record_shapes=True) as prof:
                with record_function("model_inference"):
                    predicted_colors = model(vertex_uvs)

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else: predicted_colors = model(vertex_uvs)


        predicted_colors = predicted_colors.cpu().numpy().reshape(h, w, 3)*255
        predicted_colors = predicted_colors.astype(dtype=np.uint8)
        Image.fromarray(predicted_colors).save(f"reconstructed/{name}{z}x.png")
        print(f"✓ Reconstructed {name} with {z}x zoom saved: {name}{z}x.png")

def reconstruct_fourier(model, device, height, width, z, bench = False, name = "reconstructed"):
    model.eval()
    h, w = z * height, z * width
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    test_indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
    test_indices = fourier_encoding(test_indices)
    vertex_uvs = torch.from_numpy(test_indices).float().to(device)
    predicted_colors = model(vertex_uvs)

    predicted_colors = predicted_colors.cpu().detach().numpy().reshape(h, w, 3) * 255
    predicted_colors = predicted_colors.astype(dtype=np.uint8)
    Image.fromarray(predicted_colors).save(f"reconstructed/{name}{z}x.png")
    print(f"✓ Reconstructed {name} with {z}x zoom saved: {name}{z}x.png")

def fourier_encoding(X: np.ndarray[np.ndarray[np.float32]], freqs = 2):
    x_coords = X[:,0]
    y_coords = X[:,1]
    result = np.zeros((4*freqs, X.shape[0]))
    for i in range(freqs):
        freq = 1 << i  # Frequencies: 1, 2, 4, 8
        index = i*4
        result[index] = np.sin(2 * np.pi * freq * y_coords)
        result[index + 1] = np.cos(2 * np.pi * freq * y_coords)
        result[index + 2] = np.sin(2 * np.pi * freq * x_coords)
        result[index + 3] = np.cos(2 * np.pi * freq * x_coords)
    return result.T