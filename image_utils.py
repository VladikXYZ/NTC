import os

import cv2
import numpy as np


def load_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if len(img.shape) >= 3:
        if img.shape[2] >=4: img = img[:, :, :3]
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

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

def save_image(name, img,colorspace = "rgb"):
    if img.dtype.kind != "u" and img.dtype.kind != "f": raise TypeError(f"img must be unsigned integer or float, not {img.dtype}")
    result = img
    if img.dtype.kind == "f": result = (img*255).astype(np.uint8)
    result = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
    temp = name.split("_")[0]
    path = f"converted/{temp}_{result.dtype.kind}{result.dtype.itemsize*8}{colorspace}.png"
    cv2.imwrite(path, result)

