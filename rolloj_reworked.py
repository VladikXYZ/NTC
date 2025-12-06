import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision.transforms import ToTensor, ToPILImage
import numpy as np
from tqdm import tqdm
import os


if __name__ == "__main__":
    img = Image.open("aa.png").convert('RGB')