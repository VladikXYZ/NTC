import pathlib as pth
import os
import time

import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

import FMN
import MLP
import preprocessing as prep
import numpy as np


if __name__ == "__main__":
    img_path: os.PathLike = pth.Path("data/aa.png")
    og, test_indices = prep.image_init(img_path)
    og = prep.load_image(img_path)
    all_X, all_y = [], []
    for scale in [1, 2, 3, 5, 7, 11]:
        X, y = prep.upsample(og, scale)
        all_X.append(X)
        all_y.append(y)
    X = np.vstack(all_X)
    y = np.vstack(all_y)

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

    # model = MLP.ColorMLP().to(device)
    model = FMN.NeuralDetailMapper().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # scaler = GradScaler(str(device))
    print("Training starts")
    start = time.time()
    # with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #                  schedule=torch.profiler.schedule(wait=1, warmup=1, active=6),
    #                  record_shapes=True, profile_memory=True) as prof:
    for epoch in range(10):
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
        print(f'Epoch {epoch}, Loss: {loss.item():.6f}, Epoch time: {time.time() - epoch_start:.2f}s')
    # print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    h, w = og.shape[:2]
    file_name = img_path.stem
    prep.reconstruct_image(model, device, h, w, name=file_name)
    prep.reconstruct_image(model, device, h, w, 32, bench=False, name=file_name)