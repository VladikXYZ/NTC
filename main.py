import os
from pathlib import Path
import image_utils as iu

# for x in [4,8]:
#     path = Path(f"data/r{x}.png")
#     iu.full_copy(path)

og = iu.float_conv(iu.load_image("converted/r4_Y.png"))
gen = iu.float_conv(iu.load_image("converted/r4_Ygen.png"))
snr, psnr, ssim, lpips = iu.calculate_metrics(og, gen, True)
print(f"PSNR: {psnr:.2f} dB\nSNR:  {snr:.2f} dB\nSSIM: {ssim:.3f}\nLPIPS: {lpips:.3f}")
