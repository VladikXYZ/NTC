from pathlib import Path
import image_utils as iu

og_path = Path("data/r1.png")
# gen_path = Path("converted/r1u8Ygenu8Y.png")
og = iu.load_image(og_path)
iu.save_image(og_path.stem, og)
# gen = iu.load_image(gen_path)
snr, psnr, ssim, lpips = iu.calculate_metrics(iu.float_conv(og), iu.float_conv(gen), True)
# print(f"PSNR: {psnr:.2f} dB\nSNR:  {snr:.2f} dB\nSSIM: {ssim:.2f}\nLPIPS: {lpips:.2f}")
