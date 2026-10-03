# FedEFT: Entropy-Regularized Fuzzy Trust Optimization for Fair and Byzantine-Robust Federated Learning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.13+-ee4c2c.svg)](https://pytorch.org/)


Implementation of **FedEFT**, an entropy-regularized fuzzy trust optimization framework for Byzantine-robust and fair federated learning.

---

## Overview

In federated learning (FL), the central aggregation server must assign client weights every communication round under deep uncertainty: it cannot definitively distinguish adversarial (Byzantine) updates from benign non-IID distributional shifts without violating client privacy. 

**FedEFT** addresses this challenge as an optimization problem under uncertainty:
1. **Interpretable Fuzzy Trust Engine**: Translates four client behavioral statistics (directional cosine similarity, magnitude deviations, historical consistency, and variance) into a continuous, graded trust degree $T_i \in [0, 1]$ using a Type-1 / Interval Type-2 Fuzzy Inference System.
2. **Entropy-Regularized Trust Optimization**: Client aggregation weights are derived by maximizing the expected trust reward subject to a Kullback–Leibler (KL) divergence penalty against the uniform Federated Averaging prior:
   $$\max_{\mathbf{w} \in \Delta^n} \sum_{i=1}^n w_i T_i - \tau\, D_{\mathrm{KL}}(\mathbf{w} \,\|\, \mathbf{p})$$
3. **Closed-Form Exponential Tilt**: The resulting optimal weights have a closed-form solution:
   $$w_i^* = \frac{p_i \exp(T_i / \tau)}{\sum_{j=1}^n p_j \exp(T_j / \tau)}$$
   The temperature hyperparameter $\tau$ smoothly interpolates between unbiased FedAvg ($\tau \to \infty$) and conservative hard-filtering ($\tau \to 0$).
4. **Honest Client Preservation**: Coupled with adaptive median-norm clipping, FedEFT ensures benign but unusual clients with small sample counts are never completely excluded (ensuring fairness and low variance across heterogeneous client populations).

---

## Repository Structure

```text
├── fedhift/                   # Core Python Library
│   ├── aggregators.py         # FedEFT, FLAME, Bulyan, DnC, TrimmedMean, FedAvg, learned trust
│   ├── attacks.py             # Byzantine attacks (Sign-Flip, Label-Flip, IPM, ALIE, Backdoors)
│   ├── data.py                # Dataset loaders & non-IID Dirichlet / quantity partitioners
│   ├── fl.py                  # Federated simulation loop & orchestration engine
│   ├── fuzzy.py               # Fuzzy inference engine, membership functions & rule bases
│   ├── metrics.py             # Accuracy, fairness (Jain index, min-accuracy), Byzantine recall
│   └── models.py              # CNN, ResNet-18, and MLP architectures
├── calib/                     # Calibration outputs and fitted trust parameters
│   └── scorers.json           # Calibration weights for linear, logistic, and MLP scorers
├── fodm/                      # Fuzzy visualization scripts
│   └── fig_fuzzy.py           # Plots membership functions and fuzzy trust surfaces
├── tables_fodm/               # LaTeX table templates and fragments
├── tests/                     # Test suite
│   ├── test_units.py          # 45 unit tests for core math & aggregators
│   └── test_fodm.py           # Integration tests for end-to-end simulation
├── bench_scaling.py           # Server computational cost & runtime benchmarks
├── calibrate_scorers.py       # Calibration driver for baseline parametric trust scorers
├── make_all.sh                # Script to execute analysis and table generation
├── make_aux_tables.py         # Generates auxiliary LaTeX tables (fairness, cost, sensitivity)
├── make_fodm.py               # Generates main evaluation tables and statistics
├── requirements.txt           # Python dependencies
├── run_experiments.py         # Main multi-process experimental campaign runner
├── significance.py            # Paired sign-flip permutation tests & Holm-Bonferroni correction
└── sketch_error.py            # Count-sketch gradient compression approximation analysis
```

---

## Installation & Setup

### Prerequisites
- Python 3.10 or higher
- PyTorch (CPU or CUDA)

### 1. Clone the Repository
```bash
git clone https://github.com/sudharshanbabup/Fuzzy_Federated_Learning.git
cd Fuzzy_Federated_Learning
```

### 2. Create Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

### 4. Run Verification Tests
```bash
python tests/test_units.py
python tests/test_fodm.py
```

---

## Running Federated Training Experiments

To execute training runs and evaluate aggregation rules under various Byzantine attacks:

### 1. Trust Scorer Calibration (Optional)
```bash
python calibrate_scorers.py
```

### 2. Run Main Experiment Suite
```bash
python run_experiments.py --suite main_v6 --workers 4
```

### 3. Run All Experiment Suites
```bash
python run_experiments.py --suite all --workers 4
```

### Datasets
- **Fashion-MNIST & CIFAR-10**: Automatically downloaded and preprocessed by `torchvision` into `data/`.
- **Federated EMNIST (FEMNIST)**: Download `fed_emnist.tar.bz2` from [Google TFF](https://storage.googleapis.com/tff-datasets-public/fed_emnist.tar.bz2) and extract `fed_emnist_train.h5` and `fed_emnist_test.h5` into `data/`.

---

## Implemented Defense Baselines & Attacks

### Aggregation Rules
- **FedAvg** (McMahan et al.)
- **Coordinate-wise Median & Trimmed Mean** (Yin et al.)
- **Krum & Multi-Krum** (Blanchard et al.)
- **Bulyan** (Guerraoui et al.)
- **FLAME** (Nguyen et al.)
- **DnC** (Divide and Conquer, Mustafa & Panta)
- **FedEFT** (Proposed Entropy-Regularized Fuzzy Trust Optimization)

### Byzantine Attacks
- **Random Gaussian Noise**
- **Sign-Flipping Attack**
- **Label-Flipping Attack**
- **Inner Product Manipulation (IPM)**
- **A Little Is Enough (ALIE)**
- **Targeted Backdoor Trigger Injection**
- **Adaptive Knowledge-Aware Evasion**

---


