import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import mark_inset, inset_axes
from qecsim.models.toric import ToricCode
from qecsim.models.generic import BitFlipErrorModel, DepolarizingErrorModel, BiasedDepolarizingErrorModel, BitPhaseFlipErrorModel
warnings.filterwarnings("ignore")


def plot_err(ax, codes, error_model, labels=['MWPM', 'SUNetQD', 'Enhanced_SUNetQD'], filters=['MWPM']):
    plt_shape = ['^', 's', 'o']
    plt_line = {'MWPM': '--', 'SUNetQD': '-', 'Enhanced_SUNetQD': '-.'}

    def find_npz_files(directory, filters):
        npz_files = []
        for root, _, files in os.walk(directory):
            root = root.replace("\\", "/")
            for label in labels:
                if label+".npz" in files:
                    # 使用正则表达式检查路径是否匹配过滤字符串中的任意一个
                    if not filters or any(re.search(f, root) for f in filters):
                        npz_files.append(
                            (os.path.basename(os.path.dirname(root)), label, os.path.join(root, label+".npz")))
        return npz_files

    for i, my_code in enumerate(codes):
        base_path = os.path.join(my_code.label, error_model.label, "model")
        npz_files = find_npz_files(base_path, filters)

        j = 1
        for parent_dir, label, npz_path in npz_files:
            print(my_code.label, label, parent_dir)
            data = np.load(npz_path)
            ax.plot(data['x'], 1-data['y'], plt_shape[j]+(plt_line[label] if label in plt_line else '-'),
                    label="{} {}".format(my_code.label, label))
            j -= 1
            # if "True" in parent_dir:
            #     ax.plot(data['x'], 1-data['y'], plt_shape[i]+"-",
            #              label="{} {}".format(my_code.label, "w/ Attention"))
            # else:
            #     ax.plot(data['x'], 1-data['y'], plt_shape[i]+"--",
            #              label="{} {}".format(my_code.label, "w/o Attention"))
            print(f"{data['y']}\n")


def plot_loss(ax, df):
    # Define models with more consistent naming and color scheme
    models = [
        {
            'column': "Toric 5x5/Depolarizing/model/low_decoder/32_32_5_4_(1, 2)_(True, True)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': '^',
            'linestyle': '--',
            'label': 'Toric 5x5 w/ Attention',
            'color': '#1F77B4'  # Consistent color scheme
        },
        {
            'column': "Toric 5x5/Depolarizing/model/low_decoder/32_32_5_4_(1, 2)_(False, False)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': '^',
            'linestyle': '-',
            'label': 'Toric 5x5 w/o Attention',
            'color': '#FF7F0E'
        },
        {
            'column': "Toric 7x7/Depolarizing/model/low_decoder/32_32_5_4_(1, 2, 2)_(True, True, True)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': 's',
            'linestyle': '--',
            'label': 'Toric 7x7 w/ Attention',
            'color': '#2CA02C'
        },
        {
            'column': "Toric 7x7/Depolarizing/model/low_decoder/32_32_5_4_(1, 2, 2)_(False, False, False)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': 's',
            'linestyle': '-',
            'label': 'Toric 7x7 w/o Attention',
            'color': '#D62728'
        },
        {
            'column': "Toric 9x9/Depolarizing/model/low_decoder/32_32_5_4_(1, 2, 2)_(True, True, True)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': 'o',
            'linestyle': '--',
            'label': 'Toric 9x9 w/ Attention',
            'color': '#9467BD'
        },
        {
            'column': "Toric 9x9/Depolarizing/model/low_decoder/32_32_5_4_(1, 2, 2)_(False, False, False)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': 'o',
            'linestyle': '-',
            'label': 'Toric 9x9 w/o Attention',
            'color': '#8C564B'
        },
    ]

    # Plotting with potential error bars
    for model in models:
        column = model['column']
        min_column = column + '__MIN'
        max_column = column + '__MAX'

        if column in df.columns:
            ax.plot(
                df['Step'],
                df[column],
                marker=model['marker'],
                linestyle=model['linestyle'],
                color=model['color'],
                label=model['label'],
                markevery=5
            )

    # Create inset axes with absolute sizing
    axins = inset_axes(
        ax,
        width="50%",  # 宽度为原图的50%
        height="30%",  # 高度为原图的30%
        loc='center right',
        bbox_to_anchor=(0, -0.05, 0.9, 0.9),  # 1表示完全靠右，0.5垂直居中
        bbox_transform=ax.transAxes,
    )
    # Plot zoomed region
    zoomed_steps = df['Step'] > df['Step'].max() * 0.8
    for model in models:
        column = model['column']
        if column in df.columns:
            axins.plot(
                df['Step'][zoomed_steps],
                df[column][zoomed_steps],
                marker=model['marker'],
                linestyle=model['linestyle'],
                color=model['color'],
                markevery=5
            )

    # Customize inset axes
    # axins.set_yscale('log')
    # 隐藏子图坐标轴
    axins.set_xticks([])
    axins.set_yticks(np.arange(0.08, 0.083, 0.001))

    # axins.set_title('Zoomed Region', fontsize=10)

    # Add rectangle to show zoomed region
    ax.indicate_inset_zoom(axins, edgecolor="black", alpha=0.5)

    mark_inset(ax, axins, loc1=3, loc2=4, fc="none", ec='0.5')


def plot_transfer(ax, df):
    models = [
        # {
        #     'column': "1_Toric 7x7/Depolarizing/model/low_decoder/32_32_5_4_(1, 2)_(True, True)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
        #     'marker': '^',
        #     'linestyle': '--',
        #     'label': 'Toric 7x7 transfer from 5x5',
        #     'color': '#1F77B4'
        # },
        {
            'column': "1_Toric 9x9/Depolarizing/model/low_decoder/32_32_5_4_(1, 2)_(True, True)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': '^',
            'linestyle': '-',
            'label': 'Toric 9x9 transfer from 5x5',
            'color': '#FF7F0E'
        },
        # {
        #     'column': "Toric 7x7/Depolarizing/model/low_decoder/32_32_5_4_(1, 2)_(True, True)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
        #     'marker': 's',
        #     'linestyle': '--',
        #     'label': 'Toric 7x7 train from scratch',
        #     'color': '#2CA02C'
        # },
        {
            'column': "Toric 9x9/Depolarizing/model/low_decoder/32_32_5_4_(1, 2)_(True, True)/1_{0.02: 100000, 0.025: 100000, 0.03: 100000, 0.035: 100000} - avg loss",
            'marker': 's',
            'linestyle': '-',
            'label': 'Toric 9x9 train from scratch',
            'color': '#1F77B4'
            # 'color': '#9467BD'
        },
    ]

    for model in models:
        column = model['column']
        min_column = column + '__MIN'
        max_column = column + '__MAX'

        if column in df.columns:
            ax.plot(
                df['Step'],
                df[column],
                marker=model['marker'],
                linestyle=model['linestyle'],
                color=model['color'],
                label=model['label'],
                markevery=5
            )
    ax.hlines(y=0.07935, xmin=0, xmax=250000,
              color='#2CA02C', linestyle='--', linewidth=1)


if __name__ == "__main__":
    codes = [ToricCode(*size) for size in [(5, 5), (7, 7), (9, 9)]]
    my_error_model = DepolarizingErrorModel()
    # my_error_model = BiasedDepolarizingErrorModel(5, 'Y')
    # my_error_model = BiasedDepolarizingErrorModel(50, 'Y')
    # my_error_model = BitPhaseFlipErrorModel()
    labels = ['MWPM', 'SUNetQD']
    filters = ['.*MWPM.*/1_.*', '.*low_decoder.*/1_.*']

    plt.rcParams.update({
        'font.size': 22,
        'axes.labelsize': 24,
        'axes.titlesize': 20,
        'figure.figsize': (14, 9),  # Slightly larger figure
    })

    _, ax = plt.subplots()

    # df = pd.read_csv('result.csv')
    # plot_loss(ax, df)

    # df = pd.read_csv('1.csv')
    # plot_transfer(ax, df)
    # df = pd.read_csv('2.csv')
    # plot_transfer(ax, df)

    # axins = inset_axes(
    #     ax,
    #     width="40%",  # 宽度为原图的50%
    #     height="50%",  # 高度为原图的30%
    #     loc='center right',
    #     bbox_to_anchor=(-0.05, 0, 0.95, 0.95),  # 1表示完全靠右，0.5垂直居中
    #     bbox_transform=ax.transAxes,
    # )

    plot_err(ax, codes, my_error_model, labels=labels, filters=filters)
    ax.set_title(my_error_model.label+" and Measurement Error")
    ax.set_xlabel('physical error rate')
    ax.set_ylabel('logical error rate')

    # ax.set_title(
    #     'Loss Comparison for Different Toric Code Configurations')
    # ax.set_title("Transfer Learning of SUNetQD on " + my_error_model.label+" and Measurement Error")
    # ax.set_xlabel('Training Steps')
    # ax.set_ylabel('Cross-Entropy Loss')

    # ax.yaxis.set_major_formatter(ticker.ScalarFormatter())
    # ax.yaxis.set_minor_locator(ticker.LogLocator(subs='all'))

    ax.legend(
        # loc='upper right',
        frameon=True,
        framealpha=0.7,
        fancybox=True,
        # shadow=True
    )
    plt.tight_layout()

    # plt.savefig(f'./outputs/{my_error_model.label} {" ".join(labels)}.pdf')
    # print(
    #     f'successfully saved the figure in outputs/{my_error_model.label} {" ".join(labels)}.pdf')
    plt.show()
    # plt.savefig(f'outputs/1.pdf')
    # print("successfully saved the figure in 1.pdf")
