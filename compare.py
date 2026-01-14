import argparse
import numpy as np
from PIL import Image
import os


def calculate_absolute_difference(file_name):
    og = [file_name + str(x) for x in [1, 2, 4]]
    rec = [f"{file_name}8_{str(x)}k" for x in [1, 2, 4]]
    outputs = [file_name + str(x) + "diff" for x in [1, 2, 4]]
    for o, r, out in zip(og, rec, outputs):
        img1_path, img2_path, output_path = f"data/{o}.png", f"reconstructed/{r}.png", f"reconstructed/{out}.png"

        # 1. Load Images
        print(f"Loading {img1_path}...")
        img1 = Image.open(img1_path).convert('RGB')

        print(f"Loading {img2_path}...")
        img2 = Image.open(img2_path).convert('RGB')

        # 3. Convert to float32 for math (prevents uint8 overflow/wrap-around)
        arr1 = np.array(img1, dtype=np.float32)
        arr2 = np.array(img2, dtype=np.float32)

        # 4. Calculate Absolute Difference
        # Formula: |Pixel_A - Pixel_B|
        diff_arr = np.abs(arr1 - arr2)

        # 5. Calculate Statistics (Mean Absolute Error)
        mae = np.mean(diff_arr)
        print(f"\n--- Statistics ---")
        print(f"Mean Absolute Error (MAE): {mae:.4f} (Avg pixel deviation)")
        print(f"Max Difference: {np.max(diff_arr):.4f}")

        # 6. Save Result
        # Clip to valid range [0, 255] and convert to byte
        diff_img = Image.fromarray(diff_arr.astype(np.uint8))
        diff_img.save(output_path)
        print(f"✓ Difference image saved to: {output_path}")