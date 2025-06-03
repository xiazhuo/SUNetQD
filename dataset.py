import os
import pickle
import numpy as np

import torch
from torch.utils.data import DataLoader, Dataset

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode, ToricMWPMDecoder
from qecsim.models.generic import DepolarizingErrorModel, BiasedDepolarizingErrorModel, BitPhaseFlipErrorModel


def load_data(data_path, file_name):
    file_path = os.path.join(data_path, file_name)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    return []


def save_data(data, data_path, file_name, overwrite=True):
    os.makedirs(data_path, exist_ok=True)
    file_path = os.path.join(data_path, file_name)
    if not overwrite:
        old_data = load_data(data_path, file_name)
        data = old_data + data
    with open(file_path, 'wb') as f:
        pickle.dump(data, f)


class MyDataset(Dataset):
    def __init__(self, data1, data2, targets):
        self.data1 = torch.stack(data1)
        self.data2 = torch.stack(data2) if data2 is not None else None
        self.targets = torch.stack(targets)

    def __len__(self):
        return len(self.data1)

    def __getitem__(self, index):
        if self.data2 is not None:
            return self.data1[index], self.data2[index], self.targets[index]
        return self.data1[index], self.targets[index]


def get_recovery(code, error, syndrome):
    _, _, d = code.n_k_d
    recovery = ToricMWPMDecoder().decode(code, syndrome)
    mwpm_logic_err = pt.bsp(recovery ^ error, code.logicals.T)
    if np.all(mwpm_logic_err == 0):
        if pt.bsf_wt(recovery) >= pt.bsf_wt(error):
            recovery = error
    else:
        recovery = error
    recovery = recovery.reshape(2, -1)
    recovery = recovery[0] + recovery[1]*2
    return recovery, mwpm_logic_err


def generate_data(code, noise_model, n_measure, err_prob):
    _, _, d = code.n_k_d

    real_syn, recovery, error, mwpm_logic_err = [], [], [], []
    err = np.zeros((2*2*d*d, ), dtype=int)
    for _ in range(n_measure):
        err ^= noise_model.generate(code, err_prob)
        syn = pt.bsp(err, code.stabilizers.T)
        rec, logic_err = get_recovery(code, err, syn)
        error.append(err.reshape(2, 2, d, d))
        real_syn.append(syn.reshape(2, d, d))
        recovery.append(rec.reshape(2, d, d))
        mwpm_logic_err.append(logic_err)

    error = np.stack(error)
    real_syn = np.stack(real_syn)
    recovery = np.stack(recovery)
    mwpm_logic_err = np.stack(mwpm_logic_err)

    return error, real_syn, recovery, mwpm_logic_err


def generate_dataset(code, noise_model, n_measure, size_per_error, measure_err_factor=1, train=True):
    all_errors, all_real_syndromes, all_noise_syndromes, all_recoveries, all_mwpm_logic_errs = [], [], [], [], []
    assert measure_err_factor in [1, 2, 3]

    for err_prob, data_size in size_per_error.items():
        data_path = os.path.join(code.label, noise_model.label, "train" if train else "test", "_".join(
            str(s) for s in [n_measure, err_prob]))
        os.makedirs(data_path, exist_ok=True)

        # 加载现有数据
        errors = load_data(data_path, 'errors.pt')
        real_syndromes = load_data(data_path, 'real_syndromes.pt')
        noise_syndromes = {1: [], 2: [], 3: []}
        if n_measure > 1:
            noise_syndromes[measure_err_factor] = load_data(
                data_path, 'noise_syndromes_'+str(measure_err_factor)+'.pt')
        else:
            noise_syndromes[measure_err_factor] = load_data(
                data_path, 'real_syndromes.pt')
        recoveries = load_data(data_path, 'ref_recoveries.pt')
        mwpm_logic_errs = load_data(data_path, 'mwpm_logic_errs.pt')
        assert len(errors) == len(real_syndromes) == len(noise_syndromes[measure_err_factor]) == len(
            recoveries) == len(mwpm_logic_errs)

        remain_data_size = data_size - len(errors)
        if remain_data_size > 0:
            for cnt in range(remain_data_size):
                if cnt % 1000 == 0:
                    print("data generate: {:.1f}% for err {}".format(
                        cnt*100/remain_data_size, err_prob))
                error, real_syn, recovery, mwpm_logic_err = generate_data(
                    code, noise_model, n_measure, err_prob)

                if n_measure > 1:
                    for factor in [1, 2, 3]:
                        mask = (np.random.rand(*real_syn.shape)
                                < err_prob/factor).astype(int)
                        # Perfect measurements in last round to ensure even parity
                        mask[-1] = 0
                        noise_syn = (real_syn ^ mask)
                        noise_syndromes[factor].append(torch.tensor(
                            noise_syn, dtype=torch.int64))
                else:
                    noise_syndromes[measure_err_factor].append(
                        torch.tensor(real_syn, dtype=torch.int64))

                errors.append(torch.tensor(
                    error, dtype=torch.int64))
                real_syndromes.append(torch.tensor(
                    real_syn, dtype=torch.int64))
                recoveries.append(torch.tensor(
                    recovery, dtype=torch.int64))
                mwpm_logic_errs.append(torch.tensor(
                    mwpm_logic_err, dtype=torch.int64))
                cnt += 1
            save_data(errors, data_path, 'errors.pt')
            save_data(real_syndromes, data_path, 'real_syndromes.pt')
            save_data(recoveries, data_path, 'ref_recoveries.pt')
            save_data(mwpm_logic_errs, data_path, 'mwpm_logic_errs.pt')
            if n_measure > 1:
                for factor in [1, 2, 3]:
                    save_data(noise_syndromes[factor], data_path,
                              'noise_syndromes_'+str(factor)+'.pt')

        if train:
            indexs = np.random.choice(
                len(errors), data_size, replace=False)
        else:
            indexs = np.arange(data_size)
        all_errors.extend([errors[i] for i in indexs])
        all_real_syndromes.extend([real_syndromes[i] for i in indexs])
        all_noise_syndromes.extend(
            [noise_syndromes[measure_err_factor][i] for i in indexs])
        all_recoveries.extend([recoveries[i] for i in indexs])
        all_mwpm_logic_errs.extend([mwpm_logic_errs[i] for i in indexs])
    return all_errors, all_real_syndromes, all_noise_syndromes, all_recoveries, all_mwpm_logic_errs


if __name__ == '__main__':
    my_code = ToricCode(5, 5)
    my_noise_model = DepolarizingErrorModel()
    # my_noise_model = BitPhaseFlipErrorModel()
    # my_noise_model = BiasedDepolarizingErrorModel(50, 'Y')
    test_errs = {round(prob, 3): 10 **
                 4 for prob in np.arange(0.012, 0.044, 0.004)}
    n_measure = 5
    measure_err_factor = 1
    folder_path = os.path.join(
        my_code.label, my_noise_model.label, "model", "MWPM_decoder", str(measure_err_factor)+"_"+str(test_errs))
    os.makedirs(folder_path, exist_ok=True)

    test_probs = []
    test_mwpm_acc = []
    print(my_code.label, my_noise_model.label)
    for test_err in test_errs:
        errors, real_syndromes, noise_syndromes, recoveries, mwpm_logic_errs = generate_dataset(my_code, my_noise_model, n_measure,
                                                                                                {test_err: test_errs[test_err]}, measure_err_factor, train=False)
        # print(errors[0].shape, real_syndromes[0].shape, noise_syndromes[0].shape,
        #       recoveries[0].shape, mwpm_logic_errs[0].shape)
        mwpm_acc = 0
        for i in range(len(mwpm_logic_errs)):
            if torch.all(mwpm_logic_errs[i][-1] == 0):
                mwpm_acc += 1
        mwpm_acc /= len(mwpm_logic_errs)
        test_probs.append(test_err)
        test_mwpm_acc.append(mwpm_acc)
        print("prob: {}, mwpm_acc: {}".format(test_err, mwpm_acc))

        # error = np.stack(errors)
        # print(error.reshape(*error.shape[0:2], -1).shape)
        # error = error[:, :, 0] + error[:, :, 1]*2
        # print(error.shape)

    np.savez(os.path.join(folder_path, "MWPM.npz"),
             x=test_probs,
             y=test_mwpm_acc)
