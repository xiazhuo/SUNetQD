import os
import numpy as np
from time import time

import torch
from torch.utils.data import DataLoader

from qecsim import paulitools as pt
from qecsim.models.toric import ToricCode, ToricMWPMDecoder
from qecsim.models.generic import BitFlipErrorModel, DepolarizingErrorModel

from qsu import qsu
from dataset import generate_dataset
from network import Net


def test(model, dataloader, device, folder_path, code, show_detail=False):
    def map_rule(x):
        rule = ['I', 'X', 'Z', 'Y']
        return rule[x]

    model = model.to(device)
    max_pred_num = 12
    step_acc = [0.0]*max_pred_num
    start = time()
    print("Testing started:")

    with torch.no_grad():
        model.eval()
        for error, err_synd, target in dataloader:
            error = error.cpu().numpy()
            err_synd = err_synd.to(device)
            target = target.to(device)

            origin_synd = err_synd.flatten().cpu().numpy()
            pred_recovery = np.zeros_like(error)

            if show_detail:
                qsu.print_pauli('\n\n\nerror:\n{}'.format(
                    code.new_pauli(error)))

            for i in range(max_pred_num):
                # if i // 3 == 0:
                #     file_name = "weights_" + str(0.13) + ".pth"
                if i // 3 == 0:
                    file_name = "weights_" + str(0.09) + ".pth"
                elif i // 3 == 1:
                    file_name = "weights_" + str(0.05) + ".pth"
                else:
                    file_name = "weights_" + str(0.01) + ".pth"
                if i % 3 == 0:
                    model.load_state_dict(torch.load(
                        os.path.join(folder_path, file_name)))
                output = model(err_synd.to(torch.float32))
                new_pred = output.argmax(dim=1)
                new_pred = new_pred.flatten().tolist()
                pred_recovery ^= pt.pauli_to_bsf(
                    ''.join(list(map(map_rule, new_pred))))
                if show_detail:
                    qsu.print_pauli('pred:\n{}'.format(
                        code.new_pauli(pred_recovery)))
                    qsu.print_pauli('new error:\n{}'.format(
                        code.new_pauli(pred_recovery ^ error)))

                pred_synd = pt.bsp(pred_recovery, code.stabilizers.T)
                syn = origin_synd ^ pred_synd
                if np.all(syn == 0):
                    if np.all(pt.bsp(pred_recovery ^ error, code.logicals.T) == 0):
                        step_acc[i] += 1
                        if show_detail:
                            print("succuss!")
                    break

                err_synd = torch.tensor(syn.reshape(
                    err_synd.shape), device=device)

    step_acc = [step / len(dataloader.dataset) for step in step_acc]
    now = time() - start
    print("step_acc: {}, cost time: {:.0f}m {:.0f}s".format(
        step_acc, now // 60, now % 60))
    return sum(step_acc)


if __name__ == "__main__":
    codes = [ToricCode(*size) for size in [(7, 7), (9, 9), (11, 11), (13, 13)]]
    my_error_model = BitFlipErrorModel()
    my_error_probabilities = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    # my_error_model = DepolarizingErrorModel()
    # my_error_probabilities = [0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

    assert torch.cuda.is_available()
    device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
    test_data_size = 10**3
    weight = [1.0] + [1.0]*(1 if my_error_model.label == "Bit-flip" else 3)
    model = Net(6, 12, num_class=len(weight))

    for my_code in codes:
        print(my_code.label, my_error_model.label)
        acc_list = []
        folder_path = os.path.join(my_code.label, my_error_model.label)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        for my_error_probability in my_error_probabilities:
            dataset = generate_dataset(
                my_code, my_error_model, my_error_probability, data_size=test_data_size, mode="test")
            dataloader = DataLoader(dataset)
            acc = test(model, dataloader, device, folder_path, my_code)
            acc_list.append(acc)
            print("probs:{:.2f}, acc:{:.4f}".format(
                my_error_probability, acc_list[-1]))

        np.savez(os.path.join(folder_path, 'U-Net.npz'),
                 x=np.array(my_error_probabilities), y=np.array(acc_list))
