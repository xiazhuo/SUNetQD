import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode, ToricMWPMDecoder


class MyDataset(Dataset):
    def __init__(self, errors, data, targets):
        self.errors = torch.stack(errors)
        self.data = torch.stack(data)
        self.targets = torch.stack(targets)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        x = self.errors[index]
        y = self.data[index]
        z = self.targets[index]
        return x, y, z


def get_recovery(code, error, syndrome):
    recovery = ToricMWPMDecoder().decode(code, syndrome)
    if np.all(pt.bsp(recovery ^ error, code.logicals.T) == 0):
        if pt.bsf_wt(recovery) > pt.bsf_wt(error):
            recovery = error
    else:
        recovery = error
    return recovery


def generate_dataset(code, noise_model, error_probability, data_size, n_data_aug=2, mode="train"):
    _, _, d = code.n_k_d
    errors = []
    syndromes = []
    recoveries = []

    cnt = 0
    while cnt < data_size:
        error = noise_model.generate(code, error_probability)
        syndrome = pt.bsp(error, code.stabilizers.T)
        recovery = get_recovery(code, error, syndrome)

        if mode == "train" and pt.bsf_wt(error) == 0:
            assert error_probability > 0
            continue

        recovery = recovery.reshape(2, 2, d, d)
        recovery = recovery[0] + recovery[1]*2

        origin_err = error.reshape(2, 2, d, d)
        origin_syn = syndrome.reshape(2, d, d)
        origin_rec = recovery.reshape(2, d, d)

        if n_data_aug == 1 or mode == "test":
            errors.append(torch.tensor(
                origin_err.reshape(-1), dtype=torch.int64))
            syndromes.append(torch.tensor(origin_syn, dtype=torch.int64))
            recoveries.append(torch.tensor(origin_rec, dtype=torch.int64))
            cnt += 1
            continue

        for _ in range(n_data_aug):
            shift_x = np.random.randint(0, d)
            shift_y = np.random.randint(0, d)

            tmp_err = np.roll(origin_err, shift=shift_x, axis=-2)
            tmp_syn = np.roll(origin_syn, shift=shift_x, axis=-2)
            tmp_rec = np.roll(origin_rec, shift=shift_x, axis=-2)
            error = np.roll(tmp_err, shift=shift_y, axis=-1)
            syndrome = np.roll(tmp_syn, shift=shift_y, axis=-1)
            recovery = np.roll(tmp_rec, shift=shift_y, axis=-1)
            errors.append(torch.tensor(error.reshape(-1), dtype=torch.int64))
            syndromes.append(torch.tensor(syndrome, dtype=torch.int64))
            recoveries.append(torch.tensor(recovery, dtype=torch.int64))
        cnt += n_data_aug
    return MyDataset(errors, syndromes, recoveries)


# if __name__ == '__main__':
