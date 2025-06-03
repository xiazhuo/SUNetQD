# SUNetQD: Self-attention U-Net Decoder for Toric Codes

[![Python Version](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


Official PyTorch implementation of the paper ["Self-attention U-Net decoder for toric codes"](https://doi.org/10.1103/xmp9-bh84) (SUNetQD), accepted to Physical Review Applied.


## :mag: Overview
The Self-attention U-Net Quantum Decoder (SU-NetQD) is proposed for decoding toric codes, outperforming Minimum Weight Perfect Matching (MWPM) decoders, especially in circuit-level noise environments. SU-NetQD achieves lower logical error rates and an increased code threshold with noise bias, reaching a high threshold of 0.231 for extremely biased noise. The key innovation lies in combining low-level and high-level decoders. Transfer learning allows scalability for different code distances.

## :construction_worker: TODO
<font color="red">**We are currently organizing all the code and plan to add more features.**</font>
- [x] `network.py`: Core SU-Net model definition
- [x] `dataset.py`: Script for automatic dataset generation
- [x] `train.py`: Training script for SUNetQD
- [x] `test.py`: Evaluation script for trained models
- [x] `plot.py`: Script for plotting results
- [ ] Implement training & testing code for no-measurement-noise scenarios
- [ ] Provide pre-trained model weights for no-measurement-noise scenarios
- [ ] Refactor code for improved readability and maintainability
- [ ] Add command-line argument support for all scripts (dataset generation, training, testing, plotting)
- [ ] Improve Windows compatibility for folder/file naming conventions
- [ ] Provide a wider range of pre-trained models for various code distances and noise settings

## 🛠️ Getting Started

### 1. Clone the Repository
```sh
git clone https://github.com/xiazhuo/SUNetQD.git
cd SUNetQD
```

### 2. Install Dependencies
It's recommended to use a Conda environment:
```sh
conda create -n sunetqd python=3.10
conda activate sunetqd
```
Then install the required packages:
```sh
pip install -r requirements.txt
```
Key dependencies include:
- `qecsim==1.0b9`
- `torch==2.7.0` (CUDA support is optional, e.g., `+cu118`)

### 3. Running the Code
**Important Note**: Currently, parameters for dataset generation, training, testing, and plotting (e.g., code distance, error rates, epochs, model paths) are hardcoded within the Python scripts. You will need to modify the respective .py files to change these settings. Future versions will include command-line arguments.

**(a) (Optional) Fix Folder Names for Windows Compatibility**
If you generate data/models on Linux and want to use them on Windows, you might need to run:

```sh
python fix_foldername.py
```
This script addresses issues with characters like ':' in folder names generated on Linux.

**(b) Plot Results**
Modify parameters within `plot.py` (e.g., `my_error_model`), then run:

```
python plot.py
```
This script generates plots visualizing the decoder's performance.

**(c) Test the model with the provided weights**
Modify parameters within `test.py` (e.g., `my_code`, `my_noise_model`, `low_err_probs_list), then run:

```sh
python test.py
```
This will generate test data and evaluation results, such as logical error rates.

**(d) Train the Model by yourself**
Modify parameters within `train.py` (e.g. `learning_rate`, `epochs`), then run:

```sh
python train.py
```
Trained model weights will be saved, within the corresponding `Toric XxX/` folder.


### 4. Key hyperparameters

```
Toric 5x5:
Depolarizing    Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True))
                low_err_probs_list=[0.12, 0.03]

Bias(5, 'Y')    Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True, ))
                low_err_probs_list=[0.12, 0.03]

Bias(50, 'Y')   Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True, ))
                low_err_probs_list=[0.18, 0.06]

Bit-phase-flip  Net(32, 32, n_classes=4, ch_mults=(1, 2, ),
                    use_attn=(True, True, ))
                low_err_probs_list=[0.15, 0.12]


Toric 7x7:
Depolarizing    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))
                low_err_probs_list=[0.15, 0.12]

Bias(5, 'Y')    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))
                low_err_probs_list=[0.15, 0.06]

Bias(50, 'Y')   Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))
                low_err_probs_list=[0.15, 0.06]

Bit-phase-flip  Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))
                low_err_probs_list=[0.15, 0.12]


Toric 9x9:
Depolarizing    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))
                low_err_probs_list=[0.17, 0.12, 0.15]  

Bias(5, 'Y')    Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))
                low_err_probs_list=[0.18, 0.06, 0.12, 0.04]

Bias(50, 'Y')   Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                    use_attn=(True, True, True))
                low_err_probs_list=[0.15, 0.06, 0.04]

Bit-phase-flip  Net(32, 32, n_classes=4, ch_mults=(1, 2, 2),
                   use_attn=(True, True, True))
                low_err_probs_list=[0.15, 0.12]
```

## :pushpin: Citation
If you find SUNetQD useful for your research, please cite our paper:

```bibtex
 @article{xmp9-bh84,
   title = {Self-attention U-Net decoder for toric codes},
   author = {Zhang, Wei-Wei and Xia, Zhuo and Zhao, Wei and Pan, Wei and Shi, Haobin},
   journal = {Phys. Rev. Appl.},
   pages = {--},
   year = {2025},
   month = {Jun},
   publisher = {American Physical Society},
   doi = {10.1103/xmp9-bh84},
   url = {[https://link.aps.org/doi/10.1103/xmp9-bh84](https://link.aps.org/doi/10.1103/xmp9-bh84)}
 }
```

## Acknowledgements
This work relies on the [qecsim](https://github.com/qecsim/qecsim) library for quantum error correction simulation and [PyTorch](https://github.com/pytorch/pytorch) for deep learning.
