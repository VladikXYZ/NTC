import img_utils as iu
import metrics as m



og = iu.load_image("data/r1.png")
quant = iu.uint8_conv(og)
snr, psnr, ssim, lpips = m.calculate_metrics(iu.float_conv(og), iu.float_conv(quant))
print(f"PSNR: {psnr:.2f} dB\nSNR:  {snr:.2f} dB\nSSIM: {ssim:.2f}\nLPIPS: {lpips:.2f}")
Yog = iu.Y_conv(og)
Yquant = iu.Y_conv(quant)
iu.save_image("YCoCg/r1Yog.png", Yog)
iu.save_image("YCoCg/r1Yquant.png", Yquant)