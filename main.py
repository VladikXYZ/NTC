import os
from pathlib import Path
import image_utils as iu

# for f in sorted(os.listdir("data/")):
#     print(f)
#     path = Path(f"data/{f}")
#     iu.full_copy(path)
# for x in [1,2]:
#     path = Path(f"data/r{x}.png")
#     iu.Y_copy(path, True)
for i in [1,2,4]:
    og = iu.float_conv(iu.load_image(f"converted/r{i}_u8Y.png"))
    gen = iu.float_conv(iu.load_image(f"converted/r{i}gen_u8Y.png"))
    snr, psnr, ssim, lpips = iu.calculate_metrics(og, gen, True)
    print(f"PSNR: {psnr:.2f} dB\nSNR:  {snr:.2f} dB\nSSIM: {ssim:.3f}\nLPIPS: {lpips:.3f}")
