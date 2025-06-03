import os
import wandb
import argparse
import numpy as np
from time import time

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode
from qecsim.models.generic import BitFlipErrorModel, DepolarizingErrorModel, BiasedDepolarizingErrorModel, BitPhaseFlipErrorModel

from dataset import MyDataset, generate_dataset, save_data, load_data
from network import SAUNet, logic_Net, bsp_torch


def wandb_init(disable=False):
    name = os.path.join(folder_path, model.file_dir, str(
        measure_err_factor)+"_"+str(train_errs))
    wandb.init(
        mode="disabled" if disable else "online",
        project="SUNetQD",
        entity="chirtee",
        name=name,
        reinit=True,
        tags=["v7"],
        config={
            "code": my_code.label,
            "n_measure": n_measure,
            "n_classes": n_classes,
            "error_modle": my_noise_model.label,
            "measure_err_factor": measure_err_factor,
            "size_per_err": train_errs,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "epochs": epochs,
            "channels_2d": channels_2d,
            "channels_3d": channels_3d,
            "ch_mults": ch_mults,
            "use_attn": use_attn
        }
    )


def train(model, dataloader, device, folder_path, lr=0.001, epochs=50):
    model.load_exist_state(folder_path, measure_err_factor,
                           train_errs, device=device)
    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss() if model.n_classes == 1 else nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    # scheduler = optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=epochs, eta_min=1e-4)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, mode='min', factor=0.1, patience=10)

    best_val_loss = float('inf')
    start = time()
    print("training on device:", device)
    print("Training started:")
    for epoch in range(epochs):
        model.train()
        epoch_loss = []

        for i, (data1, data2, targets) in enumerate(dataloader):
            optimizer.zero_grad()
            if isinstance(model, SAUNet):
                noise_syn = data1.to(torch.float32).to(device)
                targets = targets[:, -1]
                if model.n_classes == 1:
                    targets = targets.to(torch.float32).to(device)
                else:
                    targets = targets.to(device)
                outputs = model(noise_syn)
            else:
                noise_syn = data1.to(torch.float32).to(device)
                recovery = data2.to(torch.float32).to(device)
                targets = targets.to(device)
                outputs = model(noise_syn, recovery)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            epoch_loss.append(loss.item())
            wandb.log(
                {"loss": loss.item(), "lr": optimizer.param_groups[0]['lr']})
            if i % 99 == 0:
                now = time() - start
                print("iter {} in epoch {}: loss {:.4f}, cost time {:.0f}m {:.0f}s".format(
                    i, epoch, loss.item(), now // 60, now % 60))
                model.save_state(folder_path, measure_err_factor,
                                 train_errs, file_name="model_lst.pt")

        epoch_loss = sum(epoch_loss) / len(epoch_loss)
        # scheduler.step()
        wandb.log({"avg loss": epoch_loss})

        if epoch_loss < best_val_loss:
            best_val_loss = epoch_loss
            model.save_state(folder_path, measure_err_factor,
                             train_errs, file_name="model.pt")
        print("average loss over epoch {}: {:.4f}".format(
            epoch + 1, epoch_loss))
    return


if __name__ == "__main__":
    my_code = ToricCode(15, 15)
    _, _, d = my_code.n_k_d
    measure_err_factor = 1
    # my_noise_model = DepolarizingErrorModel()
    # train_errs = {0.02: 10**5, 0.025: 10**5, 0.03: 10**5, 0.035: 10**5}
    # my_noise_model = BiasedDepolarizingErrorModel(5, 'Y')
    # train_errs = {0.02: 10**5, 0.025: 10**5, 0.03: 10**5, 0.035: 10**5}
    my_noise_model = BiasedDepolarizingErrorModel(50, 'Y')
    # train_errs = {0.025: 10**5, 0.03: 10**5, 0.035: 10**5, 0.04: 10**5}
    # my_noise_model = BitPhaseFlipErrorModel()
    # train_errs = {0.025: 10**5, 0.03: 10**5, 0.035: 10**5, 0.04: 10**5}
    # train_errs = {0.03: 1*10**5, 0.035: 1*10**5}
    train_errs = {0.17: 2*10**5}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    learning_rate = 0.005
    epochs = 150

    channels_2d, channels_3d = 32, 32
    n_measure = 1   # !Modify
    n_classes = 4
    ch_mults = (1, 2, ) if d == 5 else (1, 2, 2)
    use_attn = (True, True, ) if d == 5 else (True, True, True)

    folder_path = os.path.join(
        my_code.label, my_noise_model.label, "model")
    os.makedirs(folder_path, exist_ok=True)
    print(folder_path)

    low_decoder = SAUNet(channels_2d, channels_3d, n_measure, n_classes,
                         ch_mults, use_attn)
    high_decoder = None
    errors, real_syndromes, noise_syndromes, recoveries, _ = generate_dataset(my_code, my_noise_model, n_measure,
                                                                              train_errs, measure_err_factor, train=True)
    dataset = MyDataset(noise_syndromes, errors, recoveries)
    dataloader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=5)

    model = low_decoder if high_decoder is None else high_decoder
    wandb_init(disable=True)
    train(model, dataloader, device, folder_path,
          lr=learning_rate, epochs=epochs)
    wandb.finish()
