# PI-LLM_for_LOCA

# PI-LLM for LOCA Transient Thermal-Hydraulic Prediction

This repository provides the implementation of **PI-LLM**, a physics-informed large language model framework for transient thermal-hydraulic parameter prediction during loss-of-coolant accidents (LOCA). The method combines the temporal representation capability of large language models with physics-informed residual constraints derived from the two-fluid six-equation model.

PI-LLM is designed as a surrogate modeling framework for RELAP5-generated LOCA transient trajectories. It integrates contrastive temporal embedding, multi-scale wavelet convolution, self-supervised reconstruction, and physics-informed regularization to improve the prediction of safety-related thermal-hydraulic variables.

## Main Components

- **PI-LLM backbone**: large-language-model-based temporal representation learning for multivariate thermal-hydraulic sequences.
- **SaCE**: contrastive temporal embedding for capturing abrupt LOCA-induced changes around key transient moments.
- **MSW-CNN**: multi-scale wavelet convolutional feature extraction for trend-dominant and transient-dominant signal components.
- **S2T**: self-supervised signal reconstruction module for enhancing physical-state representation learning.
- **Physics-informed loss**: residual constraints based on mass, momentum, and energy conservation equations.
- **Evaluation metrics**: MAE, RMSE, R2, and RRMSE.

## Repository Structure

```text
LOCA_Pinns/
├── data/                         # RELAP5-derived LOCA datasets
├── figures_results/              # Generated figures and experimental results
├── LLM_llama3/                   # LLM-related files
├── llm_weights/                  # Local LLM weights
├── models/
│   ├── pi_llm/
│   │   ├── msw_cnn.py            # Multi-scale wavelet CNN module
│   │   ├── physics.py            # Physics-informed residual loss
│   │   ├── pi_llm.py             # PI-LLM model definition
│   │   ├── s2t.py                # Self-supervised S2T module
│   │   └── sace.py               # SaCE contrastive embedding module
│   ├── PINN_add.py               # PINN-related implementation
├── savepoints/                   # Model checkpoints
├── utils/
│   ├── metrics.py                # Evaluation metrics
│   ├── plots.py                  # Visualization utilities
│   ├── read_data_exp5.py         # Data loading and preprocessing
│   ├── seq_dataset_exp5.py       # Sequence dataset construction
│   └── txtlist.py                # File list utilities
├── download_llm.py               # LLM weight download script
├── main_exp5.py                  # Baseline/experiment entry script
└── main_pi_llm.py                # Main entry script for PI-LLM
```

## Environment Setup

The code was developed and tested with Python 3.9/3.10 and PyTorch 2.0.1. A CUDA-capable GPU is recommended.

Create a virtual environment:

```bash
conda create -n pillm python=3.9 -y
conda activate pillm
```

Install PyTorch:

```bash
pip install torch==2.0.1+cu128 torchvision==0.15.2+cu128 torchaudio==2.0.2+cu128 --index-url https://download.pytorch.org/whl/cu128
```

If the `cu128` wheel is not available for your platform, install the CUDA wheel that matches your local driver and CUDA runtime from the official PyTorch package index.

Install other dependencies:

```bash
pip install numpy pandas scipy scikit-learn matplotlib tqdm einops pywavelets transformers accelerate h5py
```

## LLM Weights

Place the required LLM weights under:

```text
llm_weights/
```

If the repository is configured to download the backbone automatically, run:

```bash
python download_llm.py
```

For Hugging Face models that require authentication, log in before downloading:

```bash
huggingface-cli login
```

## Dataset Preparation

The model uses RELAP5-generated LOCA transient data as reference solutions. The monitored variables include thermal-hydraulic parameters such as pressure, void fraction, gas/liquid phase velocities, gas/liquid phase densities, friction-related variables, and gas/liquid internal energies.

Place the processed dataset files under:

```text
data/
```

The data loading and sequence construction pipeline is implemented in:

```text
utils/read_data_exp5.py
utils/seq_dataset_exp5.py
```

The complete RELAP5-generated LOCA dataset may not be included in this repository because nuclear-engineering accident data are subject to institutional data-security review. Users can place approved RELAP5-derived data with the same format under `data/` and run the training pipeline directly.

## Running PI-LLM

After preparing the environment, LLM weights, and dataset, run:

```bash
python main_pi_llm.py
```

The script trains and evaluates the PI-LLM framework. Generated checkpoints and experimental outputs are saved to:

```text
savepoints/
figures_results/
```

## Baseline Experiments

Baseline models such as PINN and KAN are implemented under `models/`. The baseline experiment entry can be run with:

```bash
python main_exp5.py
```

## Outputs and Evaluation

The main evaluation metrics are:

- **MAE**: mean absolute error
- **RMSE**: root mean square error
- **R2**: coefficient of determination
- **RRMSE**: relative root mean square error

These metrics are used to compare PI-LLM with baseline models and to evaluate prediction accuracy across different pipeline locations and physical variables.

## Reproducibility Notes

For controlled comparison, all models should be trained and evaluated using the same:

- training/testing split
- input variables and prediction targets
- normalization strategy
- optimizer setting
- training epochs and early-stopping criterion
- evaluation metrics

Due to the computational cost of LLM-based training, multi-GPU training is recommended. In the reported experiments, training was conducted on a multi-GPU server, while inference was evaluated using one or two GPUs.

## Citation

If this repository is useful for your research, please cite:

```bibtex
@article{pi_llm_loca,
  title   = {Physics-informed large language model with contrastive temporal embedding for transient thermal-hydraulic prediction in loss-of-coolant accidents},
  author  = {Yufei Xie and Co-authors},
  year    = {2026}
}
```

