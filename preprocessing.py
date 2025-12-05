import numpy as np
from PIL import Image
from scipy.ndimage import zoom

def sampling(img: np.ndarray):
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

def pipeline(image_path: str, scales: list[float]):
    img = Image.open(image_path).convert('RGB')
    img_array = np.array(img, dtype=np.float32) / 255.0
    og, test_indices = sampling(img_array)
    all_X, all_y = [], []
    for scale in scales:
        X, y = upsample(og, scale)
        all_X.append(X)
        all_y.append(y)
        # print(f"Scale {scale}: X={X.shape}, y={y.shape}")
    X = np.vstack(all_X)  # (total_samples, input_features)
    y = np.vstack(all_y)  # (total_samples, 3)
    return X,y,test_indices, og