import os

import cv2
from lpips import LPIPS
import numpy as np
import torch


def load_image(image_path):
    if not os.path.exists(image_path): raise FileNotFoundError(f"{image_path} does not exist")
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) >= 3:
        if img.shape[2] >=4: img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def tensor_float_load(image_path):
    img = float_conv(uint8_conv(load_image(image_path)))
    tensor = torch.from_numpy(img)
    return tensor

def uint8_conv(img):
    if img.dtype.kind != "u": raise TypeError(f"img must be unsigned integer not {img.dtype}")
    result = img >> 8*(img.dtype.itemsize-1)
    return np.uint8(result)

def float_conv(img):
    if img.dtype.kind != "u": raise TypeError(f"img must be unsigned integer not {img.dtype}")
    result = img/(np.power(256, img.dtype.itemsize)-1)
    return result.astype(np.float32)

def Y_conv(img):
    Y = YCoCg_conv(img)[:,:,0]
    return Y

def YCoCg_conv(img):
    if img.dtype.kind != "u": raise TypeError(f"img must be unsigned integer not {img.dtype}")
    r, g, b = img[:,:,0], img[:,:,1], img[:,:,2]
    Y = np.right_shift(r,2) + np.right_shift(g,1) + np.right_shift(b,2)
    Co = np.right_shift(r,1) - np.right_shift(b,1)
    Cg = np.right_shift(g,1) - np.right_shift(r,2) - np.right_shift(b,2)
    return np.stack((Y, Co, Cg), axis=-1)

def save_image(name, img,colorspace = "RGB", generated=False):
    if img.dtype.kind != "u" and img.dtype.kind != "f": raise TypeError(f"img must be unsigned integer or float, not {img.dtype}")
    result = img.squeeze()
    if result.dtype.kind == "f": result = (result*255).astype(np.uint8)
    if len(result.shape) >= 3: result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    temp = name.split("_")[0]
    if generated: temp += "gen"
    path = f"converted/{temp}_{result.dtype.kind}{result.dtype.itemsize*8}{colorspace}.png"
    cv2.imwrite(path, result)


def calculate_metrics(img1, img2, calculate_lpips=False):
    if img1.shape != img2.shape: raise ValueError(f"img1 and img2 have different shapes: {img1.shape}, {img2.shape}")
    if img1.dtype != img2.dtype != np.float32: raise ValueError(f"img1 and img2 must be np.float32: {img1.dtype}, {img2.dtype}")
    mse = np.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf'), float('inf'), 1, 0


    # Calculate SNR
    signal_power = np.mean(img1 ** 2)
    snr = 10 * np.log10(signal_power / mse)

    # Calculate PSNR
    max_pixel = 1
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))

    # Calculate SSIM
    C1 = (0.01 * max_pixel) ** 2
    C2 = (0.03 * max_pixel) ** 2
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.cov(img1.flatten(), img2.flatten())[0, 1]

    num = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    den = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2)
    ssim = num / den

    # Calculate LPIPS
    lpips = -1
    if calculate_lpips:
        loss_fn_alex = LPIPS(net="vgg")
        def prepare_tensor(img):
            if len(img.shape) == 2:  # Handle grayscale
                img = np.stack([img] * 3, axis=-1)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return t*2 - 1

        img1_t = prepare_tensor(img1)
        img2_t = prepare_tensor(img2)
        dist = loss_fn_alex(img1_t, img2_t)
        lpips = dist.item()

    return snr, psnr, ssim, lpips

def full_copy(img_path):
    res = load_image(img_path)
    save_image(img_path.stem, res, "RGB", False)
    YCoCg = YCoCg_conv(res)
    save_image(img_path.stem, YCoCg, "YCoCg", False)
    Y = YCoCg[:,:,0]
    save_image(img_path.stem, Y, "Y", False)
    res = uint8_conv(res)
    save_image(img_path.stem, res, "RGB", False)
    YCoCg = YCoCg_conv(res)
    save_image(img_path.stem, YCoCg, "YCoCg", False)
    Y = YCoCg[:, :, 0]
    save_image(img_path.stem, Y, "Y", False)