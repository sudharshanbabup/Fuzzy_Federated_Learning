# Fuzzy Federated Learning (FedEFT)

Reference implementation of **FedEFT**: Byzantine-robust and fair federated learning aggregation with Interval Type-2 (IT2) fuzzy trust allocation and a cumulative influence ledger.

---

## Repository Structure

| Path | Description |
|---|---|
| `fedhift/` | Core library: fuzzy inference engine, cumulative influence ledger, aggregation algorithms (FedEFT and baselines), attack strategies, models, and simulation engine |
| `calibrate_scorers.py` | Fits and calibrates trust scoring models using calibration federations |
| `run_experiments.py` | Main experimental execution script across Byzantine attacks, data heterogeneity, and parameter sweeps |
| `run_ledger_study.py` | Executes multi-round cumulative influence ledger simulations |
| `analyze_ledger.py` | Parses and analyzes cumulative influence ledger logs and metrics |
| `bench_scaling.py` | Measures server computational overhead and scaling with respect to model dimensions |
| `significance.py` | Statistical hypothesis testing (paired sign-flip permutation tests with Holm correction) |
| `make_fodm.py` | Generates summary tables and evaluation metrics |
| `make_all.sh` | Shell script running the end-to-end data processing and evaluation pipeline |
| `results_ledger.tar.gz` | Compressed archive containing ledger evaluation run records |
| `tests/` | Unit tests for aggregation rules, fuzzy inference logic, and capped allocation |

---

## Setup & Installation

### Prerequisites
- Python 3.10+
- PyTorch (CPU or CUDA)

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

---

## Execution Guide

### 1. Run Unit Tests
```bash
python -m pytest tests -q
```

### 2. Calibrate Trust Scorers
```bash
python calibrate_scorers.py
```

### 3. Run Experimental Suites
```bash
# Run all experimental suites (or specify individual suite names)
python run_experiments.py --suite all --workers 4

# Run cumulative influence ledger study (specify worker count)
python run_ledger_study.py 4

# Run server aggregation scaling benchmark
python bench_scaling.py
```

### 4. Process and Generate Results
```bash
# Extract precomputed ledger records if not already present
[ -d results_ledger ] || tar -xzf results_ledger.tar.gz

# Run full evaluation pipeline (tables and metrics)
bash make_all.sh

# Or run individual analysis steps:
python make_fodm.py
python analyze_ledger.py results_ledger generated tables_fodm
python fodm/fig_fuzzy.py
```
