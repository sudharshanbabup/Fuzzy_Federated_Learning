# FedEFT — Entropy-Regularized Fuzzy Trust Aggregation for Federated Learning

Reference implementation and full experimental campaign for the paper
*Fuzzy Trust-Guided Entropy-Regularized Aggregation for Fair and
Byzantine-Robust Federated Learning* (`../tex/main.tex`, Springer Nature
format).

Everything runs on CPU. The whole campaign was produced on a two-core cloud
instance; no GPU is required and none was used.

---

## Layout

```
codes_1/
  fedhift/
    fuzzy.py         interval type-2 engine: Gaussian uncertain-sigma sets,
                     heterogeneity-driven FOU, 16-rule TSK base, exact
                     centre-of-sets type reduction, entropic weight allocation
    aggregators.py   FedAvg, FedAvg+clip, coordinate median, trimmed mean,
                     Multi-Krum, RFA, FLTrust, the KL-cos and KL-linear
                     controls, and the proposed FedEFT rule
    attacks.py       label flip, noisy label, sign flip, Gaussian, scaling,
                     ALIE, IPM, and the white-box adaptive adversary built
                     against the four trust statistics
    data.py          Dirichlet non-IID partitioning + per-client matched test sets
    models.py        SmallCNN (Fashion-MNIST), CIFARCNN with GroupNorm
    metrics.py       accuracy, balanced accuracy and macro F1 per client,
                     Jain fairness index, worst-decile, exact AUROC
    fl.py            federated simulator and per-run bookkeeping
  tests/test_units.py   33 unit tests (see "Auditing" below)
  run_experiments.py    campaign driver; every run is cached as one JSON
  snstyle.py            the single figure style for the manuscript: TeX Gyre
                        Heros at 9 pt, the Okabe-Ito palette, the class's true
                        text width, and the shape-preserving smoothing used for
                        every line plot
  make_sn.py            every figure and table of the manuscript, into
                        figures_sn/ and tables_sn/
  make_pipeline_fig.py  the three-layer schematic of Fig. 1
  bench_scaling.py      server aggregation cost against model dimension
  plotstyle.py          the earlier two-column figure style, kept because
                        make_figures.py still uses it
  make_figures.py       most paper figures and LaTeX tables
  make_v6.py            the main and fairness tables, the component
                        decomposition and the adaptive-attack rho sweep
  make_tables_extra.py  the FOU-comparison and sensitivity tables
  sketch_error.py       measures the sketched cosine against exact computation
  summarize.py          prints every number quoted in the paper
  significance.py       exact permutation tests, confidence intervals, Cohen's
                        d_z and Holm correction behind every comparative claim
  verify_trust.py       trust dynamics, and predicted vs measured Byzantine
                        weight mass across fourteen instrumented runs
  diagnostics.py        per-attack inspection of the antecedent statistics, and
                        the two-stage heterogeneity estimate vs. a naive MAD
  results/<suite>/*.json
  figures/  tables/  logs/
```

## Reproducing

```bash
pip install torch torchvision numpy matplotlib --index-url https://download.pytorch.org/whl/cpu
python run_experiments.py --suite all --workers 2      # ~8 h on two cores
bash run_v6_all.sh                                     # the later suites
python run_experiments.py --suite adaptive_full --workers 2
python bench_scaling.py                                # dimension sweep, no training
python sketch_error.py                                 # sketch accuracy, Table 9
python verify_trust.py                                 # trust dynamics
python make_pipeline_fig.py                            # Fig. 1
python make_sn.py                                      # every other figure and table
python make_tables_extra.py                            # tab_fou, tab_sens
python tests/test_units.py                             # 33 unit tests
cd ../tex && bash build.sh                             # builds main.pdf
```

Datasets download automatically to `data/` on first use.

The driver writes one JSON per run, keyed by an MD5 of its configuration, and
skips runs whose output already exists. A suite can therefore be interrupted and
resumed at no cost; `run_rest.sh` wraps the driver in a retry loop for exactly
that reason.

### Suites

| suite | runs | what it establishes |
|---|---|---|
| `main_v6` | 370 | ten rules x eight conditions x five seeds, including both non-fuzzy controls and the adaptive attack |
| `main_fmnist` | 132 | the earlier seven-rule grid, kept because the ablation and sweep suites are paired against it |
| `ablation_ipm` | 32 | the same under inner-product manipulation, plus the nine-way ablation |
| `noniid` | 192 | Dirichlet sweep, alpha in {0.05, 0.1, 0.3, 1.0}, eight rules, six seeds |
| `byzfrac` | 24 | adversary fraction in {0, 0.1, 0.3, 0.4}, single seed |
| `ablation` | 40 | one component removed at a time, two attacks, two seeds |
| `fou` | 40 | interval type-2 vs. type-1 across five alpha, four seeds |
| `fou_temp` | 48 | the same across four temperatures |
| `sensitivity` | 40 | T, kappa and sketch dimension |
| `adaptive_rho` | 45 | the white-box attack's alignment budget rho, three rules, three seeds |
| `adaptive_full` | 48 | the trust-aware and clip-aware variants against eight rules, three seeds |
| `main_cifar` | 105 | CIFAR-10 confirmation study, three seeds |
| `cifar_v6` | 30 | the two non-fuzzy controls on CIFAR-10, three seeds |
| `clip_a005` | 8 | isolates the contribution of norm clipping at alpha = 0.05 |

Total: 1,154 training runs, plus two measurements that train nothing, the
synthetic dimension sweep in `results/scaling.json` (`bench_scaling.py`) and the
sketch-error measurement in `results/sketch_error.json` (`sketch_error.py`).

`run_v6_all.sh` runs the four suites added in the second revision in order,
each wrapped in a retry loop.

## The method in one page

Per round the server receives `{Delta_k}` and computes, on a fixed random
coordinate sketch:

1. **alignment** — cosine to the *spherical* geometric median of the cohort
   (normalising before taking the median denies a magnitude-inflating adversary
   any leverage over the reference itself);
2. **peer agreement** — mean cosine to the `ceil(m/2)-1` nearest peers, which a
   colluding minority cannot manufacture on its own;
3. **magnitude regularity** — `exp(-|log(||Delta_k|| / median)| / 0.7)`;
4. **temporal stability** — cosine to a unit-norm exponential memory of the
   client's own past directions.

A type-1 pass over the same rule base ranks the cohort; the robust dispersion of
the alignment statistic *within the more trusted half* sets the FOU width, so an
adversary cannot inflate the estimate to buy itself leniency. Interval type-2
inference and exact type reduction give a trust degree, which is smoothed into a
reputation and turned into weights by

```
w_k = pi_k * exp(tau_k / T) / sum_j pi_j * exp(tau_j / T)
```

the closed-form maximiser of `<w, tau> - T * KL(w || pi)`. Updates are clipped to
the median norm before averaging.

`T -> infinity` recovers FedAvg exactly; `T -> 0` recovers single-client hard
selection. `tests/test_units.py` checks both limits numerically.

## Auditing

```bash
python tests/test_units.py     # or: python -m pytest tests -q
```

The 33 tests are the audit trail for the parts that are easy to get quietly
wrong:

* **type reduction** is checked against an exhaustive switch-point search on 200
  random rule bases — this caught a genuine off-by-one in an earlier iterative
  Karnik–Mendel implementation, which is why the shipped version enumerates all
  `R+1` switch points with prefix sums instead;
* **the weight allocation** is checked against 2000 random simplex points per
  trial to confirm it maximises the KL-regularised objective, and against the
  two temperature limits;
* **Proposition 3** (bounded participation distortion) is checked numerically on
  200 random instances;
* **the engine** is checked for evidence-monotonicity in all four antecedents and
  for the FOU contracting the trust spread;
* **attacks** are checked to leave honest updates untouched and to have the
  intended geometry (e.g. sign flipping reverses direction to within 1e-3);
* **FedEFT** is checked to be permutation-equivariant, to reduce exactly to
  FedAvg at high temperature, and to respect the clipping bound of Theorem 6;
* **the allocation** is checked against the ratio and total-variation bounds of
  Lemma 1 on 40 random problems;
* **the adaptive adversary** is checked to carry the cohort median norm at
  exactly the prescribed cosine to the coalition mean, to be identical across
  the coalition and reproducible across runs;
* **FedAvg+clip** is checked to coincide with FedAvg until the clip binds;
* **the balanced metrics** are checked to separate a perfect classifier from
  the majority-class predictor that plain accuracy rewards.

Two defects found by this process are worth recording, since both silently
degrade results rather than crashing:

* the coordinate sketch was originally redrawn every round, which made the
  temporal-stability statistic compare vectors living in different subspaces and
  pinned it at the neutral value;
* the ALIE attack used an unbiased standard deviation, which is `NaN` when a
  single adversary is sampled in a round and poisoned every aggregator downstream.

## Notes

* Multi-Krum is given the true expected number of adversaries and FLTrust a clean
  100-sample server root set. Both are advantages FedEFT does not receive.
* Every rule sees the identical federation, client-sampling sequence and
  adversarial set for a given seed; the adversarial draw happens even when the
  attack is `none` so that the random streams stay aligned.
* Per-client accuracy is measured on a private test set whose label mix matches
  that client's own training data, which is what makes the fairness numbers
  meaningful under label skew.
* Because the federation, sampling sequence and adversarial set are identical
  across rules for a given seed, every comparison in the paper is a *paired*
  test over matched (attack, seed) cells. `significance.py` prints them. Reading
  unpaired means here would overstate the differences.
