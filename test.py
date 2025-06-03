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


def test(denoise_decoder, low_decoders, high_decoder, dataloader, device, folder_path, prob, n_pred_list):
    save_dir = None
    if denoise_decoder is not None:
        denoised_decoder = denoise_decoder.to(device)
        denoised_decoder.eval()
        save_dir = os.path.join(folder_path, denoise_decoder.file_dir, str(
            measure_err_factor)+"_"+str(train_errs), str(prob))
    for low_decoder in low_decoders:
        low_decoder = low_decoder.to(device)
        low_decoder.eval()
    if high_decoder is not None:
        high_decoder = high_decoder.to(device)
        high_decoder.eval()

    no_syn_rate = 0.0
    origin_acc = 0.0
    better_acc = 0.0
    stabilizers = torch.tensor(
        my_code.stabilizers.T, device=device)
    logicals = torch.tensor(
        my_code.logicals.T, device=device)

    def get_pred(outputs):
        new_pred = outputs.argmax(dim=1).flatten(start_dim=1)
        return torch.concat([new_pred % 2, new_pred // 2], dim=1)

    start = time()
    pred_syndromes, pred_recoveries, SUNetQD_logic_errors = [], [], []
    print("Testing started:")
    with torch.no_grad():
        for data1, targets in dataloader:
            real_syndromes = data1[:, -1].to(device)
            noise_syndromes = data1.to(device)
            errors = targets[:, -1].flatten(start_dim=1).to(device)
            pred_recovery = torch.zeros_like(errors)

            if denoise_decoder is not None:
                outputs = denoised_decoder(noise_syndromes.to(torch.float32))
                pred_recovery ^= get_pred(outputs)
                pred_synd = bsp_torch(pred_recovery, stabilizers)
                remain_synd = real_syndromes ^ pred_synd.view(
                    real_syndromes.shape)
            else:
                remain_synd = real_syndromes

            for i, n_pred in enumerate(n_pred_list):
                low_decoder = low_decoders[i]
                for _ in range(n_pred):
                    outputs = low_decoder(remain_synd.to(torch.float32))
                    pred_recovery ^= get_pred(outputs)
                    pred_synd = bsp_torch(pred_recovery, stabilizers)
                    remain_synd = real_syndromes ^ pred_synd.view(
                        real_syndromes.shape)

            no_synd_ids = torch.nonzero(
                torch.all(remain_synd.flatten(start_dim=1) == 0, dim=1)).squeeze(1)
            real_syndromes = real_syndromes[no_synd_ids]
            no_syn_rate += len(no_synd_ids)
            pred_recovery = pred_recovery[no_synd_ids]
            errors = errors[no_synd_ids]
            logic_error = bsp_torch((pred_recovery ^ errors), logicals)
            origin_acc += torch.all(logic_error ==
                                    torch.tensor([0, 0, 0, 0]).to(device), dim=1).sum().item()

            pred_recovery = pred_recovery.view(-1, 2, 2, d, d)
            pred_recovery = pred_recovery[:, 0]+pred_recovery[:, 1]*2

            if high_decoder is not None:
                logic_error_pred = high_decoder(real_syndromes.to(
                    torch.float32), pred_recovery.to(torch.float32)).ge(0)
                better_acc += torch.all(
                    logic_error == logic_error_pred, dim=1).sum().item()
            else:
                for i in range(real_syndromes.size(0)):
                    pred_syndromes.append(real_syndromes[i].cpu())
                    pred_recoveries.append(pred_recovery[i].cpu())
                    SUNetQD_logic_errors.append(logic_error[i].cpu())

    no_syn_rate /= len(dataloader.dataset)
    origin_acc /= len(dataloader.dataset)
    better_acc /= len(dataloader.dataset)

    # if high_decoder is None and save_dir is not None:
    #     save_data(pred_syndromes, save_dir,
    #               'pred_syndromes.pt', overwrite=True)
    #     save_data(pred_recoveries, save_dir,
    #               'pred_recoveries.pt', overwrite=True)
    #     save_data(SUNetQD_logic_errors, save_dir,
    #               'SUNetQD_logic_errors.pt', overwrite=True)

    now = time() - start
    print("no_synd_rate: {:.4f}, origin_acc: {:.4f}, better_acc: {:.4f}, cost time: {:.0f}m {:.0f}s".format(
        no_syn_rate, origin_acc, better_acc, now // 60, now % 60))
    return origin_acc, better_acc


if __name__ == "__main__":
    my_code = ToricCode(5, 5)
    _, _, d = my_code.n_k_d
    measure_err_factor = 1
    my_noise_model = DepolarizingErrorModel()
    train_errs = {0.02: 10**5, 0.025: 10**5, 0.03: 10**5, 0.035: 10**5}
    # my_noise_model = BiasedDepolarizingErrorModel(5, 'Y')
    # train_errs = {0.02: 10**5, 0.025: 10**5, 0.03: 10**5, 0.035: 10**5}
    # my_noise_model = BiasedDepolarizingErrorModel(50, 'Y')
    # train_errs = {0.025: 10**5, 0.03: 10**5, 0.035: 10**5, 0.04: 10**5}
    # my_noise_model = BitPhaseFlipErrorModel()
    # train_errs = {0.025: 10**5, 0.03: 10**5, 0.035: 10**5, 0.04: 10**5}

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    batch_size = 256
    channels_2d, channels_3d = 32, 32
    n_measure = 5
    n_classes = 4
    ch_mults = (1, 2, ) if d == 5 else (1, 2, 2)
    use_attn = (True, True, ) if d == 5 else (True, True, True)

    folder_path = os.path.join(
        my_code.label, my_noise_model.label, "model")
    os.makedirs(folder_path, exist_ok=True)
    print(folder_path)

    denoise_decoder = SAUNet(channels_2d, channels_3d, n_measure, n_classes,
                             ch_mults, use_attn)
    denoise_decoder.load_exist_state(
        folder_path, measure_err_factor, train_errs, device=device, file_name="model.pt")

    low_err_probs_list = [
        # {0.2: 2*10**5},
        # {0.18: 2*10**5},
        # {0.06: 2*10**5},
        # {0.15: 2*10**5},
        # {0.06: 2*10**5},
        {0.12: 2*10**5},
        # {0.04: 2*10**5},
        {0.03: 2*10**5},
    ]
    n_pred_list = [4]*len(low_err_probs_list)
    low_decoders = []
    for err_probs in low_err_probs_list:
        low_decoder = SAUNet(channels_2d, channels_3d, 1, n_classes,
                             ch_mults, use_attn)
        low_decoder.load_exist_state(
            folder_path, measure_err_factor, err_probs, device=device)
        low_decoders.append(low_decoder)
    high_decoder = None

    test_probs, test_origin_acc, test_better_acc = [], [], []
    for prob in np.arange(0.012, 0.044, 0.004):     # 最后一轮无测量误差
        prob = round(prob, 3)
        print(my_code.label, my_noise_model.label, prob)
        test_err = {prob: 10**4}
        errors, real_syndromes, noise_syndromes, recoveries, mwpm_logic_errs = generate_dataset(my_code, my_noise_model, n_measure,
                                                                                                test_err, measure_err_factor, train=False)
        dataset = MyDataset(noise_syndromes, None, errors)
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=5)
        origin_acc, better_acc = test(
            denoise_decoder, low_decoders, high_decoder, dataloader, device, folder_path, prob, n_pred_list)
        test_probs.append(prob)
        test_origin_acc.append(origin_acc)
        test_better_acc.append(better_acc)

    print(test_probs)
    print(test_origin_acc)
    # file_dir = denoise_decoder.file_dir if high_decoder is None else high_decoder.file_dir
    # np.savez(os.path.join(folder_path, file_dir, str(measure_err_factor)+"_"+str(train_errs), "SUNetQD.npz"),
    #          x=test_probs,
    #          y=test_origin_acc)
    # np.savez(os.path.join(folder_path, file_dir, str(measure_err_factor)+"_"+str(train_errs), "Enhanced_SUNetQD.npz"),
    #          x=test_probs,
    #          y=test_better_acc)
