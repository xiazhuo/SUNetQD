import os
import numpy as np
from matplotlib import pyplot as plt

from qecsim import app
from qecsim.models.toric import ToricCode, ToricMWPMDecoder
from qecsim.models.generic import BitFlipErrorModel, DepolarizingErrorModel


def mwmp(codes, error_model, error_probability, test_data_size):
    for my_code in codes:
        refer_acc = []
        folder_path = os.path.join(my_code.label, error_model.label)
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        print(my_code.label, error_model.label)
        for error_probability in my_error_probabilities:
            data = app.run(my_code, error_model, ToricMWPMDecoder(),
                           error_probability, max_runs=test_data_size)
            refer_acc.append(1 - data["logical_failure_rate"])
            print("probs:{:.2f}, acc:{:.4f}".format(
                error_probability, refer_acc[-1]))
        np.savez(os.path.join(folder_path, 'MWMP.npz'), x=np.array(
            my_error_probabilities), y=np.array(refer_acc))


def plot_err(codes, error_model, labels=['MWMP.npz']):
    for my_code in codes:
        folder_path = os.path.join(my_code.label, error_model.label)
        for label in labels:
            data = np.load(os.path.join(folder_path, label))
            plt.plot(data['x'], data['y'], 'o-',
                     label=label+" {}".format(my_code.label))

    plt.legend()
    plt.title(error_model.label)
    plt.xlabel('error probability')
    plt.ylabel('accuracy')
    plt.show()


if __name__ == "__main__":
    codes = [ToricCode(*size) for size in [(7, 7), (9, 9), (11, 11), (13, 13)]]
    my_error_model = BitFlipErrorModel()
    my_error_probabilities = [0.01, 0.03, 0.05, 0.07, 0.09, 0.11, 0.13, 0.15]
    # my_error_model = DepolarizingErrorModel()
    # my_error_probabilities = [0.01, 0.05, 0.09, 0.13, 0.17, 0.21, 0.25]

    mwmp(codes, my_error_model, my_error_probabilities, 10**3)
    plot_err(codes, my_error_model)
