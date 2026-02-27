import os
from pathlib import Path
import image_utils as iu

# for f in sorted(os.listdir("data/")):
#     print(f)
#     path = Path(f"data/{f}")
#     iu.full_copy(path)
# iu.full_copy(Path("data/r4.png"))
og = iu.float_conv(iu.load_image("converted/r1_u16Y.png"))
gen = iu.float_conv(iu.load_image("converted/r1gen_u8Y.png"))
snr, psnr, ssim, lpips = iu.calculate_metrics(og, gen, True)
print(f"PSNR: {psnr:.2f} dB\nSNR:  {snr:.2f} dB\nSSIM: {ssim:.3f}\nLPIPS: {lpips:.3f}")
