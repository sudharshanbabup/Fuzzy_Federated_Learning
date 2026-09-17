# Fuzzy Federated Learning (FedEFT)

This repository contains the official implementation of **Entropy-Regularized Fuzzy Trust Aggregation for Federated Learning (FedEFT)**.

FedEFT provides a Byzantine-robust and fair federated learning aggregation framework designed to mitigate malicious client attacks (such as label flipping, sign flipping, Gaussian noise, scaling, ALIE, and IPM attacks) under non-IID data distributions.

---

## 📁 Repository Structure

```
.
├── fedhift/                  # Core package implementation
│   ├── __init__.py           # Package initialization
│   ├── aggregators.py        # Aggregation rules (FedAvg, Median, Trimmed Mean, Krum, RFA, FLTrust, FedEFT)
│   ├── attacks.py            # Byzantine attack implementations (Label flip, Sign flip, Noise, ALIE, IPM, etc.)
│   ├── data.py               # Dirichlet non-IID data partitioning and test set construction
│   ├── fl.py                 # Federated learning simulation driver & round bookkeeping
│   ├── fuzzy.py              # Interval Type-2 Fuzzy Logic engine (FOU estimation, TSK inference, type reduction)
│   ├── metrics.py            # Evaluation metrics (Accuracy, Jain Fairness Index, Worst-Decile, AUROC)
│   └── models.py             # Neural network architectures (SmallCNN, CIFARCNN with GroupNorm)
├── tests/                    # Unit tests suite
│   └── test_units.py         # 24 unit tests covering type reduction, weight allocation, bounds, and attacks
├── run_experiments.py        # Campaign execution driver for experimental suites
├── run_rest.sh               # Shell script to retry and complete experimental suites
├── make_figures.py           # Script to plot figure visualizations from experiment outputs
├── make_pipeline_fig.py      # Script to render system pipeline diagram
├── make_tables_extra.py      # Script to generate comparison and sensitivity tables
├── summarize.py              # Summary utility for experimental outputs
├── significance.py           # Paired statistical significance testing script
├── diagnostics.py           # Antecedent statistics and heterogeneity diagnostic inspector
└── README.md                 # Project documentation
```

---

## 🛠️ Environment Setup & Installation

### Prerequisites
- Python 3.8+
- PyTorch (CPU or GPU)

### Install Dependencies

```bash
pip install torch torchvision numpy matplotlib
```

---

## 🚀 Usage

### Running Unit Tests
Validate the implementation, fuzzy engine, aggregators, and attack models:
```bash
python -m unittest discover -s tests
# or using pytest:
# pytest tests/
```

### Running Experiments
Execute the experimental suite driver:
```bash
python run_experiments.py --suite main_fmnist --workers 2
```

Available experiment suites include:
- `main_fmnist`: Evaluation across standard aggregators and Byzantine attack scenarios on Fashion-MNIST.
- `main_cifar`: Confirmation study on CIFAR-10 dataset.
- `noniid`: Dirichlet heterogeneity sweep ($\alpha \in \{0.05, 0.1, 0.3, 1.0\}$).
- `byzfrac`: Evaluation under varying adversary fractions.
- `ablation`: Component-wise ablation study.
- `fou`: Interval Type-2 vs. Type-1 fuzzy logic comparison.
- `sensitivity`: Parameter sensitivity sweeps (temperature, sketch dimension).

### Plotting & Analysis
Generate pipeline figures, performance plots, and statistical analysis:
```bash
python make_pipeline_fig.py
python make_figures.py
python make_tables_extra.py
python summarize.py
python significance.py
```

---

## 📜 Method Overview

Per round, the server collects client updates and evaluates them across a fixed random coordinate sketch using four antecedents:
1. **Alignment**: Cosine similarity to the spherical geometric median of the cohort.
2. **Peer Agreement**: Mean cosine similarity to nearest peer updates.
3. **Magnitude Regularity**: Exponential deviation measure relative to cohort median update norm.
4. **Temporal Stability**: Cosine similarity to exponential moving average of client past directions.

An **Interval Type-2 Fuzzy Logic System (IT2-FLS)** computes client trust degrees from these antecedents, which are then converted into aggregation weights using entropy-regularized softmax weighting:

$$w_k = \frac{\pi_k \exp(\tau_k / T)}{\sum_j \pi_j \exp(\tau_j / T)}$$

---



