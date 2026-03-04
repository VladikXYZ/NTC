import cv2
import numpy as np
import torch
import torch.fft
import matplotlib.pyplot as plt

import image_utils as iu



def get_standard_fft_results(img):
    f_coeff = torch.fft.fft2(img)
    magnitude = 20 * torch.log(torch.abs(f_coeff))
    reconstruction = torch.fft.ifft2(f_coeff).real
    return magnitude, reconstruction


def get_fast_real_fft_results(img):
    f_coeff = torch.fft.rfft2(img)
    mask = torch.ones_like(f_coeff.real)
    mask[:CUT, :CUT] = 0
    mask[-CUT:, :CUT] = 0
    f_coeff *= mask
    magnitude = 20 * torch.log(torch.abs(f_coeff))
    reconstruction = torch.fft.irfft2(f_coeff, s=img.shape)
    return magnitude, img - reconstruction

def run_fft_comparison(img):
    img_tensor = torch.from_numpy(img).float()
    # mag, recon = get_fast_real_fft_results(img_tensor)
    mag, recon = get_dct(img)
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    for i, (show, cap) in enumerate(zip([img_tensor, mag, recon],["OG", "Magnitude", "Reconstruction"])):
        axes[i].imshow(show, cmap="gray" if i % 2 == 0 else "viridis")
        axes[i].set_title(cap)
        axes[i].set_axis_off()

    plt.tight_layout()
    plt.show()


CUT = 1024

def get_dct(img):
    img_float32 = np.float32(img) / 255.0
    dct_result = cv2.dct(img_float32)
    dct_result[:CUT, :CUT] = 0
    magnitude = np.abs(np.log(np.abs(dct_result)+1))
    idct_result = cv2.idct(dct_result)
    img_reconstructed = (idct_result * 255.0).astype(np.uint8)
    iu.save_image("dct", np.asarray(img-img_reconstructed)*255, "Y")
    return magnitude, img - img_reconstructed

img = iu.load_image("converted/r8_Y.png")
get_dct(img)
# run_fft_comparison(img)