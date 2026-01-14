import pathlib as pth
import os
import time

import torch
from PIL import Image
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader
import preprocessing as prep
import numpy as np
import models

import argparse
parser = argparse.ArgumentParser(description="Example script")
parser.add_argument("-img", help="input file path")
parser.add_argument("-nn", help="model")
testable = {"fmn" : models.NeuralDetailMapper, "mlp" : models.ColorMLP}


if __name__ == "__main__":
    args = parser.parse_args()
    if not args.img: args.img = "data/oa.png"
    if not args.nn: args.nn = "fmn"
    img_path: os.PathLike = pth.Path(args.img)
    og, test_indices = prep.image_init(img_path)
    og = prep.load_image(img_path)
    all_X, all_y = [], []
    for scale in [1]:
        X, y = prep.upsample(og, scale)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.vstack(all_y)
    # X = prep.fourier_encoding(X)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    print(f"Using device: {device}")

    inputs_tensor = torch.from_numpy(X).float()
    targets_tensor = torch.from_numpy(y).float()

    dataset = TensorDataset(inputs_tensor, targets_tensor)
    is_cuda = (str(device) == 'cuda')
    # print(is_cuda)
    loader = DataLoader(dataset, batch_size=max(256, 4096 * is_cuda), shuffle=True, num_workers=12 * is_cuda,
                        pin_memory=is_cuda)

    model = testable[args.nn]().to(device) #MLP.ColorMLP().to(device)
    # model = FMN.NeuralDetailMapper().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scaler = GradScaler(str(device))
    print("Training starts")
    start = time.time()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #                  schedule=torch.profiler.schedule(wait=1, warmup=1, active=6),
    #                  record_shapes=True, profile_memory=True) as prof:
    for epoch in range(50):
        epoch_start = time.time()
        for batch_inputs, batch_targets in loader:
            # prof.step()
            batch_inputs = batch_inputs.to(device, non_blocking=True)
            batch_targets = batch_targets.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            preds = model(batch_inputs)
            loss = criterion(preds, batch_targets)
            # if is_cuda:
            #     scaler.scale(loss).backward()
            #     scaler.step(optimizer)
            #     scaler.update()
            # else:
            loss.backward()
            optimizer.step()
        print(f'Epoch {epoch:03d}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')
    print(f"Training took {time.time()-start}s")
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    h, w = og.shape[:2]
    file_name = img_path.stem
    # prep.reconstruct_fourier(model, device, h, w, 1, name=file_name)
    # prep.reconstruct_fourier(model, device, h, w, 2, name=file_name)
    prep.reconstruct_image(model, device, h, w, name=file_name)
    prep.reconstruct_image(model, device, h, w, 4, bench=False, name=file_name)




# def train(model, coords, colors, device, epochs=50, batch_size=16384):
#     print(f"\n--- Starting Training on {device} ---")
#
#     # Dataset
#     dataset = TensorDataset(
#         torch.from_numpy(coords).float(),
#         torch.from_numpy(colors).float()
#     )
#
#     # Optimized DataLoader
#     use_cuda = device.type == 'cuda'
#     loader = DataLoader(
#         dataset,
#         batch_size=batch_size,
#         shuffle=True,
#         num_workers=12 if use_cuda else 0,
#         pin_memory=use_cuda
#     )
#
#     optimizer = optim.Adam(model.parameters(), lr=1e-2)
#
#     # Reduce LR on plateau for fine details
#     scheduler = optim.lr_scheduler.ReduceLROnPlateau(
#         optimizer, mode='min', factor=0.5, patience=5
#     )
#
#     criterion = nn.MSELoss()
#
#     start_time = time.time()
#     model.train()
#
#     for epoch in range(epochs):
#         total_loss = 0
#         batches = 0
#
#         for bx, by in loader:
#             bx, by = bx.to(device, non_blocking=True), by.to(device, non_blocking=True)
#
#             optimizer.zero_grad(set_to_none=True)
#             preds = model(bx)
#             loss = criterion(preds, by)
#             loss.backward()
#             optimizer.step()
#
#             total_loss += loss.item()
#             batches += 1
#
#         avg_loss = total_loss / batches
#         scheduler.step(avg_loss)
#
#         # Calculate PSNR every few epochs
#         if (epoch + 1) % 5 == 0 or epoch == 0:
#             psnr = -10 * math.log10(avg_loss) if avg_loss > 0 else 100
#             print(f"Epoch {epoch + 1:03d}/{epochs} | Loss: {avg_loss:.6f} | PSNR: {psnr:.2f}dB")
#
#     print(f"Training Time: {time.time() - start_time:.2f}s")