import torch
import torch.fft
import matplotlib.pyplot as plt

import image_utils as iu

CUT = 128

def get_standard_fft_results(img):
    f_coeff = torch.fft.fft2(img)
    magnitude = 20 * torch.log(torch.abs(f_coeff) + 1e-8)
    reconstruction = torch.fft.ifft2(f_coeff).real
    return magnitude, reconstruction


def get_fast_real_fft_results(img):
    f_coeff = torch.fft.rfft2(img)
    mask = torch.ones_like(f_coeff.real)
    mask[:CUT, :CUT] = 0
    mask[-CUT:, -CUT:] = 0
    f_coeff *= mask
    magnitude = 20 * torch.log(torch.abs(f_coeff) + 1e-8)
    reconstruction = torch.fft.irfft2(f_coeff, s=img.shape)
    return magnitude, img - reconstruction

def run_fft_comparison(img):
    img_tensor = torch.from_numpy(img).float()
    mag, recon = get_fast_real_fft_results(img_tensor)
    fig, axes = plt.subplots(1, 3, figsize=(10, 5))
    for i, (show, cap) in enumerate(zip([img_tensor, mag, recon],["OG", "Magnitude", "Reconstruction"])):
        axes[i].imshow(show, cmap="gray" if i % 2 == 0 else "viridis")
        axes[i].set_title(cap)
        axes[i].set_axis_off()

    plt.tight_layout()
    plt.show()

img = iu.load_image("converted/r1_Y.png")
run_fft_comparison(img)