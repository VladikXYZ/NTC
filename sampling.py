import scipy as scp
import numpy as np
import torch
from PIL import Image
from sympy import false
from torch.profiler import profile, record_function, ProfilerActivity


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
    samples = scp.ndimage.zoom(img, (scale, scale, 1), order=1)  # order=1 = bilinear
    h, w = samples.shape[:2]
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)  # (N, 2)
    samples = samples.reshape(-1, 3)
    return indices, samples

def reconstruct_image(model, device, height, width, z = 1, bench = False, halved = False):
    model.eval()
    h, w = z*height, z*width
    y_coords, x_coords = np.meshgrid(np.linspace(0, 1, h), np.linspace(0, 1, w))
    test_indices = np.stack([y_coords.flatten(), x_coords.flatten()], axis=1)
    with torch.no_grad():
        if halved: vertex_uvs = torch.from_numpy(test_indices).half().to(device)
        else: vertex_uvs = torch.from_numpy(test_indices).float().to(device)
        if bench:
            activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
            with profile(activities=activities, record_shapes=True) as prof:
                with record_function("model_inference"):
                    predicted_colors = model(vertex_uvs)

            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
        else: predicted_colors = model(vertex_uvs)


        predicted_colors = predicted_colors.cpu().numpy().reshape(h, w, 3)*255
        predicted_colors = predicted_colors.astype(dtype=np.uint8)
        Image.fromarray(predicted_colors).save(f"reconstructed{z}.png")
        print(f"✓ Reconstructed image with {z} zoom saved: reconstructed{z}.png")