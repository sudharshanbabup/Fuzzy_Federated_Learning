# Fuzzy Federated Learning: FedEFT

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU%20optimized-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-33%2F33%20passed-brightgreen.svg)](tests/test_units.py)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Official reference implementation and complete experimental campaign records for the research paper:  
**"Fuzzy Trust-Guided Entropy-Regularized Aggregation for Fair and Byzantine-Robust Federated Learning"**

---

## 🌟 Overview

**FedEFT** (Entropy-Regularized Fuzzy Trust Aggregation) is a Byzantine-robust and participation-fair federated learning aggregation framework. It addresses the fundamental tension between Byzantine robustness and client fairness under severe statistical heterogeneity (Dirichlet non-IID label skew).

### Key Highlights
* **Interval Type-2 (IT2) Fuzzy Reasoning Core**: Employs a 16-rule zero-order Takagi–Sugeno–Kang (TSK) rule base driven by four server-observable update statistics:
  1. **Directional Alignment**: Cosine similarity to the cohort's spherical geometric median.
  2. **Peer Agreement**: Mean cosine similarity to nearest peer updates.
  3. **Magnitude Regularity**: Exponential penalty on deviations from the cohort median norm.
  4. **Temporal Stability**: Cosine similarity to an exponential moving average of past client directions.
* **Closed-Form Entropy-Regularized Allocation**: Solves $\max_{\mathbf{w}} \langle \mathbf{w}, \boldsymbol{\tau} \rangle - T \cdot D_{\mathrm{KL}}(\mathbf{w} \parallel \boldsymbol{\pi})$, yielding:
  $$w_k = \frac{\pi_k \exp(\tau_k / T)}{\sum_j \pi_j \exp(\tau_j / T)}$$
  Smoothly interpolates between standard FedAvg ($T \to \infty$) and hard single-client selection ($T \to 0$).
* **Fast Coordinate Sketching**: Operates on a fixed $d = 2^{14}$ random coordinate sketch, keeping per-round server overhead independent of model parameter dimension $p$.
* **Full Reproducibility**: Includes all **1,154 raw experimental run records** (JSON format) across 14 experimental suites, enabling instant regeneration of all manuscript tables and figures without re-training.
* **100% CPU Compatible**: Designed to execute entirely on standard multi-core CPUs without requiring GPUs.

---

## 📁 Repository Layout

```
.
├── fedhift/                  # Core package implementation
│   ├── __init__.py           # Package exports
│   ├── aggregators.py        # FedEFT, FedAvg, FedAvg+clip, Median, Trimmed Mean, Multi-Krum, RFA, FLTrust, KL-cos
│   ├── attacks.py            # Label flip, Sign flip, Gaussian noise, Scaling, ALIE, IPM, Adaptive white-box
│   ├── data.py               # Dirichlet non-IID partitioning & per-client matched evaluation sets
│   ├── fl.py                 # Federated simulation driver & round bookkeeping
│   ├── fuzzy.py              # IT2 fuzzy inference engine (uncertain-sigma sets, EKM exact type reduction)
│   ├── metrics.py            # Accuracy, balanced accuracy, macro F1, Jain fairness index, worst-decile, AUROC
│   └── models.py             # SmallCNN (Fashion-MNIST) and CIFARCNN with GroupNorm
├── tests/
│   └── test_units.py         # 33 comprehensive unit tests (algorithmic invariants & mathematical bounds)
├── make_sn.py                # Generates all Springer Nature tables and figures from results/
├── snstyle.py                # Manuscript visual styling (Okabe-Ito palette, vector layout)
├── make_pipeline_fig.py      # Generates system pipeline architecture schematic
├── bench_scaling.py          # Synthetic dimension scaling benchmark (10^4 to 3x10^6 parameters)
├── sketch_error.py           # Sketched cosine approximation error analysis vs. full dimension
├── summarize.py              # Computes and prints every aggregate metric quoted in the paper
├── significance.py           # Paired permutation tests, exact p-values, 95% CIs, and Holm corrections
├── verify_trust.py           # Trust trajectory verification & predicted vs. measured Byzantine mass
├── diagnostics.py            # Antecedent inspection and two-stage heterogeneity estimator checks
├── run_experiments.py        # Campaign driver with configuration hashing and caching
├── run_v6_all.sh             # Shell script runner for complete revision suites
├── tables_sn/                # Generated LaTeX tables for manuscript
├── figures_sn/               # Generated publication-quality vector PDF figures
├── results/                  # 1,154 raw experimental run records (.json) across 14 suites
│   ├── main_v6/              # 370 runs: 9 aggregators x 8 conditions x 5 seeds
│   ├── main_fmnist/          # 132 runs: Fashion-MNIST baseline grid
│   ├── main_cifar/           # 105 runs: CIFAR-10 confirmation study (3 seeds)
│   ├── cifar_v6/             # 30 runs: non-fuzzy controls on CIFAR-10
│   ├── noniid/               # 192 runs: Dirichlet alpha in {0.05, 0.1, 0.3, 1.0} sweep (6 seeds)
│   ├── adaptive_full/        # 48 runs: trust-aware and clip-aware white-box adaptive attack variants
│   ├── adaptive_rho/         # 45 runs: alignment budget rho sweep
│   ├── ablation/             # 40 runs: component ablations under label and sign flipping
│   ├── ablation_ipm/         # 32 runs: component ablations under inner-product manipulation
│   ├── fou/                  # 40 runs: IT2 vs. Type-1 reduction across Dirichlet alpha
│   ├── fou_temp/             # 48 runs: IT2 vs. Type-1 reduction across temperatures
│   ├── sensitivity/          # 40 runs: sensitivity sweeps over T, kappa, and sketch dimension d
│   ├── byzfrac/              # 24 runs: Byzantine participant fraction sweep {0, 0.1, 0.3, 0.4}
│   ├── clip_a005/            # 8 runs: isolated median clipping contribution at alpha = 0.05
│   ├── scaling.json          # Synthetic dimension scaling benchmark records
│   └── sketch_error.json     # Sketched cosine approximation measurements
├── logs/                     # Execution logs and summaries
└── README.md
```

---

## 🚀 Quick Start & Installation

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/sudharshanbabupandava/Fuzzy_Federated_Learning.git
cd Fuzzy_Federated_Learning

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies (CPU PyTorch)
pip install torch torchvision numpy matplotlib scipy --index-url https://download.pytorch.org/whl/cpu
```

### 2. Run Unit Tests (Auditing)

Verify all 33 unit tests covering exact Karnik–Mendel type reduction, entropy maximization, participation bounds, and attack invariants:

```bash
python tests/test_units.py
# Expected output: 33/33 passed
```

---

## 📊 Reproducing Paper Results Without Re-training

Because all 1,154 raw execution JSONs are bundled under `results/`, all tables, figures, and statistical tests in the manuscript can be regenerated immediately:

### 1. Regenerate All Manuscript Tables and Figures
```bash
python make_sn.py
```
* Generates LaTeX tables in `tables_sn/` (`tab_main_fmnist.tex`, `tab_main_cifar.tex`, `tab_fairness.tex`, `tab_adaptive.tex`, `tab_cost.tex`, `tab_stats.tex`, etc.).
* Generates vector PDF figures in `figures_sn/` (`fig_hetero.pdf`, `fig_fairness.pdf`, `fig_ablation.pdf`, `fig_rho.pdf`, `fig_scaling.pdf`, `fig_sensitivity.pdf`, `fig_trust.pdf`).

### 2. Print All Quoted Manuscript Metrics
```bash
python summarize.py
```

### 3. Print Hypothesis Tests & Multiplicity-Corrected $p$-values
```bash
python significance.py
```
Computes exact two-sided sign-flip permutation tests, 95% confidence intervals, Cohen's $d_z$, and Holm–Bonferroni adjusted $p$-values.

### 4. Auxiliary Analyses
```bash
python sketch_error.py      # Measures cosine sketch accuracy vs. full dimension (Table 8)
python bench_scaling.py     # Measures aggregation runtime scaling across dimension p (Table 9)
python verify_trust.py      # Trust trajectory & predicted vs. measured Byzantine weight mass
python diagnostics.py       # Antecedent statistics and two-stage dispersion analysis
```

---

## 🔬 Executing Training Campaigns

To train models from scratch or extend the benchmark:

```bash
# Run a specific experimental suite (e.g. main_v6 or noniid)
python run_experiments.py --suite main_v6 --workers 2

# Or run complete revision campaign
bash run_v6_all.sh
```

Datasets (Fashion-MNIST and CIFAR-10) automatically download to `data/` upon first invocation. Every run is cached as a JSON file keyed by the MD5 hash of its configuration, allowing interrupted runs to resume seamlessly.

---

## 📖 Citation

If you use this codebase or find our work helpful in your research, please cite:

```bibtex
@article{fedeft2026fuzzy,
  title={Fuzzy Trust-Guided Entropy-Regularized Aggregation for Fair and Byzantine-Robust Federated Learning},
  author={Pandava, Sudharshan Babu and collaborators},
  journal={Scientific Reports},
  year={2026}
}
```

---

## 📄 License

This repository is distributed under the [MIT License](LICENSE).
