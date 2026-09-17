# Fuzzy Federated Learning: FedEFT

Reference implementation and experimental campaign codebase for **FedEFT (Entropy-Regularized Fuzzy Trust Aggregation for Federated Learning)**.

FedEFT is a Byzantine-robust and fair federated learning aggregation algorithm designed to defend against data and model poisoning attacks (label flip, sign flip, Gaussian noise, scaling, ALIE, IPM, and white-box adaptive attacks) under non-IID client data distributions.

---

## 📁 Repository Layout

```
.
├── fedhift/                  # Core library implementation
│   ├── __init__.py           # Package exports
│   ├── aggregators.py        # Aggregation algorithms: FedEFT, FedAvg, FedAvg+clip, Median, Trimmed Mean, Multi-Krum, RFA, FLTrust, KL-cos
│   ├── attacks.py            # Byzantine attacks: Label flip, Sign flip, Gaussian noise, Scaling, ALIE, IPM, Adaptive attack
│   ├── data.py               # Dirichlet non-IID partitioning and per-client evaluation sets
│   ├── fl.py                 # Federated learning simulation driver and round bookkeeping
│   ├── fuzzy.py              # Interval Type-2 fuzzy inference engine (uncertain-sigma sets, EKM exact type reduction)
│   ├── metrics.py            # Evaluation metrics: Accuracy, balanced accuracy, macro F1, Jain fairness index, worst-decile, AUROC
│   └── models.py             # Neural network models (SmallCNN for Fashion-MNIST, CIFARCNN with GroupNorm)
├── tests/
│   └── test_units.py         # Unit tests covering type reduction, entropic weight allocation, bounds, and attacks
├── make_sn.py                # Generates manuscript tables and vector figures from experimental records
├── snstyle.py                # Visual styling for figures (Okabe-Ito palette, vector layout)
├── make_pipeline_fig.py      # Generates system pipeline schematic
├── bench_scaling.py          # Synthetic dimension scaling benchmark (10^4 to 3x10^6 parameters)
├── sketch_error.py           # Sketched cosine approximation error analysis vs. full dimension
├── summarize.py              # Summary utility for experimental outputs
├── significance.py           # Paired permutation hypothesis tests, 95% CIs, and Holm corrections
├── verify_trust.py           # Trust trajectory verification and Byzantine weight mass measurement
├── diagnostics.py            # Antecedent statistics and heterogeneity dispersion analysis
├── run_experiments.py        # Campaign driver with configuration hashing and caching
├── run_v6_all.sh             # Shell script runner for revision experiment suites
├── tables_sn/                # Generated LaTeX tables
├── figures_sn/               # Generated vector PDF figures
├── results/                  # Complete experimental run records (.json) across all suites
│   ├── main_v6/              # 370 runs: 9 aggregators x 8 conditions x 5 seeds
│   ├── main_fmnist/          # 132 runs: Fashion-MNIST baseline grid
│   ├── main_cifar/           # 105 runs: CIFAR-10 confirmation study
│   ├── cifar_v6/             # 30 runs: controls on CIFAR-10
│   ├── noniid/               # 192 runs: Dirichlet alpha sweep (6 seeds)
│   ├── adaptive_full/        # 48 runs: trust-aware and clip-aware adaptive attack variants
│   ├── adaptive_rho/         # 45 runs: alignment budget rho sweep
│   ├── ablation/             # 40 runs: component ablations under label and sign flipping
│   ├── ablation_ipm/         # 32 runs: component ablations under inner-product manipulation
│   ├── fou/                  # 40 runs: IT2 vs. Type-1 reduction across Dirichlet alpha
│   ├── fou_temp/             # 48 runs: IT2 vs. Type-1 reduction across temperatures
│   ├── sensitivity/          # 40 runs: sensitivity sweeps over T, kappa, and sketch dimension d
│   ├── byzfrac/              # 24 runs: Byzantine participant fraction sweep
│   ├── clip_a005/            # 8 runs: isolated median clipping contribution at alpha = 0.05
│   ├── scaling.json          # Synthetic dimension scaling benchmark records
│   └── sketch_error.json     # Sketched cosine approximation measurements
├── logs/                     # Execution logs and summaries
└── README.md
```

---

## 🛠️ Environment Setup & Installation

### Requirements
* Python 3.8+
* All code runs on CPU (no GPU required)

### Setup

```bash
# Clone the repository
git clone https://github.com/sudharshanbabupandava/Fuzzy_Federated_Learning.git
cd Fuzzy_Federated_Learning

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU PyTorch)
pip install torch torchvision numpy matplotlib scipy --index-url https://download.pytorch.org/whl/cpu
```

---

## 🧪 Running Unit Tests

Run the test suite covering type reduction, entropy maximization, participation bounds, and attack invariants:

```bash
python tests/test_units.py
```

Expected output: `33/33 passed`.

---

## 📊 Generating Results, Tables, and Figures

All 1,154 execution JSON records are included under `results/`, allowing immediate generation of all outputs without re-running training:

### 1. Generate All Tables and Figures
```bash
python make_sn.py
```
* Writes LaTeX tables to `tables_sn/`
* Writes vector PDF figures to `figures_sn/`

### 2. Print Summary Statistics
```bash
python summarize.py
```

### 3. Print Statistical Significance & Hypothesis Tests
```bash
python significance.py
```
Computes paired permutation tests, 95% confidence intervals, Cohen's $d_z$, and Holm–Bonferroni adjusted $p$-values.

### 4. Auxiliary Analyses
```bash
python sketch_error.py      # Measures cosine sketch accuracy vs. full dimension
python bench_scaling.py     # Measures aggregation runtime scaling across dimension p
python verify_trust.py      # Trust trajectory & predicted vs. measured Byzantine weight mass
python diagnostics.py       # Antecedent statistics and two-stage dispersion analysis
```

---

## 🚀 Running Training Campaigns

To re-run training suites from scratch or execute new configurations:

```bash
# Run a specific suite (e.g. main_v6 or noniid)
python run_experiments.py --suite main_v6 --workers 2

# Or run complete revision campaign via shell script
bash run_v6_all.sh
```

Datasets (Fashion-MNIST and CIFAR-10) automatically download to `data/` upon first run. Every run is cached as a JSON file keyed by the MD5 hash of its configuration, allowing interrupted runs to resume without re-running completed configurations.

---

