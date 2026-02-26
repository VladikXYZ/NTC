import lpips as l
import numpy as np
import torch


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
        loss_fn_alex = l.LPIPS(net="vgg")
        def prepare_tensor(img):
            if len(img.shape) == 2:  # Handle grayscale
                img = np.stack([img] * 3, axis=-1)
            t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)
            return (t / 127.5) - 1.0

        img1_t = prepare_tensor(img1)
        img2_t = prepare_tensor(img2)
        dist = loss_fn_alex(img1_t, img2_t)
        lpips = dist.item()

    return snr, psnr, ssim, lpips
