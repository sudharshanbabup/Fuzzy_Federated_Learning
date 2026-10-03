"""Federated learning simulator."""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .aggregators import AGGREGATORS, FedHIFT, build_aggregator
from .attacks import (BACKDOOR, BACKDOOR_BOOST, BACKDOOR_FRAC, BACKDOOR_TARGET,
                      DATA_POISON, MODEL_POISON, _fixed_direction,
                      apply_model_poison, poison_labels)
from .data import FederatedData, add_trigger, load_federated
from .metrics import (auc_binary, evaluate, evaluate_detailed, jain_index,
                      worst_frac)
from .models import build_model, num_params

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@dataclass
class FLConfig:
    dataset: str = "fmnist"
    num_clients: int = 30
    clients_per_round: int = 10
    rounds: int = 50
    local_epochs: int = 1
    batch_size: int = 32
    lr: float = 0.05
    momentum: float = 0.9
    weight_decay: float = 0.0
    alpha: float = 0.5              # Dirichlet non-IID concentration
    byz_frac: float = 0.2
    attack: str = "none"
    aggregator: str = "fedavg"
    seed: int = 0
    train_subsample: int | None = 30000
    root_size: int = 100
    prox_mu: float = 0.0            # FedProx proximal coefficient
    attack_rho: float = 0.7         # alignment budget of the adaptive attack
    eval_every: int = 5
    # FedEFT hyper-parameters
    temperature: float = 0.20
    nu: float = 1.0
    beta_rep: float = 0.5
    sketch_dim: int = 16384
    type1: bool = False
    use_size_prior: bool = True
    criteria: List[str] = field(default_factory=lambda: ["align", "peer", "norm", "stab"])
    kappa: float | None = None
    tag: str = ""
    # ---- extensions of the FODM revision (defaults reproduce earlier runs) ----
    partition: str = "dirichlet"    # "dirichlet" | "quantity"
    sigma_q: float = 1.0            # log-normal spread of client sizes (quantity skew)
    attack_start: int = 0           # adversaries behave honestly up to this round
    dropout: float = 0.0            # stationary fraction of unavailable clients
    absence: float = 5.0            # mean length of an unavailability spell (rounds)
    keep_log: bool = False          # full per-round log incl. theory quantities
    scorer_params: str = "calib/scorers.json"
    rule_perturb: float = 0.0       # relative perturbation of rule consequents
    rule_seed: int = 0
    track_dir: bool = False         # log drift along the adaptive attack direction


def local_train(model, x, y, cfg: FLConfig, gen: torch.Generator,
                global_vec: torch.Tensor | None) -> torch.Tensor:
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=cfg.lr, momentum=cfg.momentum,
                          weight_decay=cfg.weight_decay)
    n = len(y)
    for _ in range(cfg.local_epochs):
        perm = torch.randperm(n, generator=gen)
        for i in range(0, n, cfg.batch_size):
            idx = perm[i:i + cfg.batch_size]
            if len(idx) < 2:
                continue
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x[idx]), y[idx])
            if cfg.prox_mu > 0 and global_vec is not None:
                v = parameters_to_vector(model.parameters())
                loss = loss + 0.5 * cfg.prox_mu * torch.sum((v - global_vec) ** 2)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
    return parameters_to_vector(model.parameters()).detach()


def perturbed_rules(eps: float, seed: int):
    """Multiply every consequent end-point by an independent factor drawn
    uniformly from [1 - eps, 1 + eps], clip to [0,1] and keep lo <= hi."""
    from .fuzzy import RULE_BASE
    r = np.random.default_rng(10007 + seed)
    out = []
    for ante, (lo, hi) in RULE_BASE:
        a = float(np.clip(lo * (1 + r.uniform(-eps, eps)), 0, 1))
        b = float(np.clip(hi * (1 + r.uniform(-eps, eps)), 0, 1))
        out.append((ante, (min(a, b), max(a, b))))
    return out


@torch.no_grad()
def attack_success_rate(model, x_bd: torch.Tensor, batch: int = 512) -> float:
    model.eval()
    if len(x_bd) == 0:
        return float("nan")
    hit = 0
    for i in range(0, len(x_bd), batch):
        hit += (model(x_bd[i:i + batch]).argmax(1) == BACKDOOR_TARGET).sum().item()
    return hit / len(x_bd)


def _theory_record(info: Dict, U: torch.Tensor, delta: torch.Tensor, lbl: List[int],
                   T: float) -> Dict:
    """Per-round quantities that appear in the bounds of Section 4."""
    w = np.asarray(info["w"], dtype=np.float64)
    pi = np.asarray(info["prior"], dtype=np.float64)
    r = np.asarray(info["tau"], dtype=np.float64)
    lab = np.asarray(lbl)
    B, H = lab == 0, lab == 1
    out = {"W_B": float(w[B].sum()), "Pi_B": float(pi[B].sum()),
           "M": float(info["med_norm"]), "phi": float(info.get("phi", 1.0))}
    if H.sum() >= 2:
        out["delta_H"] = float(r[H].max() - r[H].min())
    sc = torch.tensor(np.asarray(info["clip_scale"], dtype=np.float64), dtype=U.dtype)
    Uc = U * sc[:, None]
    wH = w[H] / max(w[H].sum(), 1e-300)
    hon_raw = (torch.tensor(wH, dtype=U.dtype)[:, None] * U[H]).sum(0)
    hon_clip = (torch.tensor(wH, dtype=U.dtype)[:, None] * Uc[H]).sum(0)
    out["clip_err"] = float(torch.linalg.norm(hon_raw - hon_clip))
    out["hon_norm"] = float(torch.linalg.norm(hon_raw))
    if B.any() and H.any():
        out["delta_unif"] = float(r[H].min() - r[B].max())
        out["delta_mean"] = float(r[H].mean() - r[B].mean())
        PiB, PiH = out["Pi_B"], 1.0 - out["Pi_B"]
        # mean-field substitution of the population trust means
        out["mf_mass"] = float(PiB / (PiB + PiH * np.exp(out["delta_mean"] / T)))
        # Corollary 1 bound, valid only when the uniform margin is positive
        out["bound_mass"] = float(PiB / PiH * np.exp(-out["delta_unif"] / T))
        out["bias"] = float(torch.linalg.norm(delta - hon_clip))
    return out


def run(cfg: FLConfig, fd: FederatedData | None = None, verbose: bool = False) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.seed + 7919)
    gen = torch.Generator().manual_seed(cfg.seed + 104729)

    if fd is None:
        fd = load_federated(cfg.dataset, cfg.num_clients, cfg.alpha, cfg.seed,
                            cfg.train_subsample, cfg.root_size,
                            cfg.partition, cfg.sigma_q)

    # The adversarial set is always drawn (even when it is unused) so that the
    # random stream that selects participants is identical across every
    # aggregation rule and attack for a given seed.
    n_byz = int(round(cfg.byz_frac * cfg.num_clients))
    drawn = set(rng.choice(cfg.num_clients, size=max(n_byz, 1), replace=False).tolist())
    malicious = drawn if (cfg.attack != "none" and n_byz > 0) else set()

    model = build_model(cfg.dataset)
    global_vec = parameters_to_vector(model.parameters()).detach().clone()
    theta0 = global_vec.clone()
    work = build_model(cfg.dataset)

    # cached poisoned labels for data-poisoning adversaries
    poisoned: Dict[int, torch.Tensor] = {}
    if cfg.attack in DATA_POISON:
        for k in malicious:
            ix = fd.client_idx[k]
            poisoned[k] = poison_labels(fd.y_train[ix], cfg.attack, fd.num_classes, gen)
    # backdoor: a fixed fraction of each adversary's samples carries the trigger
    poisoned_x: Dict[int, torch.Tensor] = {}
    x_bd = None
    if cfg.attack in BACKDOOR:
        brng = np.random.default_rng(cfg.seed + 31337)
        for k in sorted(malicious):
            ix = fd.client_idx[k]
            xk, yk = fd.x_train[ix].clone(), fd.y_train[ix].clone()
            sel = brng.choice(len(ix), size=int(round(BACKDOOR_FRAC * len(ix))), replace=False)
            sel_t = torch.as_tensor(sel, dtype=torch.long)
            xk[sel_t] = add_trigger(xk[sel_t])
            yk[sel_t] = BACKDOOR_TARGET
            poisoned_x[k], poisoned[k] = xk, yk
        nt = fd.y_test != BACKDOOR_TARGET
        x_bd = add_trigger(fd.x_test[nt])

    if cfg.aggregator.startswith("fedhift") or cfg.aggregator in ("klcos", "kllin") \
            or cfg.aggregator.startswith("tr_"):
        from .fuzzy import IT2Config
        it2 = IT2Config()
        if cfg.kappa is not None:
            it2.kappa = cfg.kappa
        if cfg.rule_perturb > 0:
            it2.rules = perturbed_rules(cfg.rule_perturb, cfg.rule_seed)
        sp = None
        if cfg.aggregator[3:] in ("linopt", "logistic", "mlp"):
            with open(os.path.join(HERE, cfg.scorer_params)) as fh:
                sp = json.load(fh)
        kw = dict(temperature=cfg.temperature, nu=cfg.nu, beta_rep=cfg.beta_rep,
                  sketch_dim=cfg.sketch_dim, use_size_prior=cfg.use_size_prior,
                  seed=cfg.seed)
        if cfg.aggregator.startswith("fedhift"):
            agg = FedHIFT(type1=cfg.type1, cfg=it2, criteria=cfg.criteria, **kw)
        elif cfg.aggregator == "tr_type1":
            agg = FedHIFT(type1=True, cfg=it2, criteria=cfg.criteria, **kw)
        elif cfg.aggregator.startswith("tr_"):
            agg = FedHIFT(scorer=cfg.aggregator[3:], scorer_params=sp, cfg=it2,
                          criteria=cfg.criteria, **kw)
        else:
            agg = build_aggregator(cfg.aggregator, **kw)
    else:
        agg = AGGREGATORS[cfg.aggregator]

    state: Dict = {"n_byz_est": max(1, int(round(cfg.byz_frac * cfg.clients_per_round))),
                   "seed": cfg.seed}
    hist = {"round": [], "acc": [], "loss": [], "asr": [], "proj": [], "drift": []}
    w_log: List[Dict] = []
    theory: List[Dict] = []
    agg_time = 0.0
    t0 = time.time()

    # client availability (churn): a two-state Markov chain per client with a
    # stationary unavailable fraction `dropout` and mean absence `absence`
    online = np.ones(cfg.num_clients, dtype=bool)
    if cfg.dropout > 0:
        arng = np.random.default_rng(cfg.seed + 555)
        p_on = 1.0 / cfg.absence
        p_off = cfg.dropout * p_on / (1.0 - cfg.dropout)
        online = arng.random(cfg.num_clients) >= cfg.dropout
    b_dir = None
    if cfg.track_dir:
        b_dir = _fixed_direction(global_vec.numel(), global_vec.dtype)

    for rnd in range(1, cfg.rounds + 1):
        if cfg.dropout > 0:
            flip = arng.random(cfg.num_clients)
            online = np.where(online, flip >= p_off, flip < p_on)
            avail = np.where(online)[0]
            if len(avail) < 3:                      # keep a minimal cohort
                avail = np.union1d(avail, arng.choice(cfg.num_clients, 3, replace=False))
            sel = rng.choice(avail, size=min(cfg.clients_per_round, len(avail)),
                             replace=False)
        else:
            sel = rng.choice(cfg.num_clients, size=cfg.clients_per_round, replace=False)
        active = rnd > cfg.attack_start
        updates: Dict[int, torch.Tensor] = {}
        sizes = []
        for k in sel:
            vector_to_parameters(global_vec.clone(), work.parameters())
            ix = fd.client_idx[k]
            xk = poisoned_x[k] if (active and k in poisoned_x) else fd.x_train[ix]
            yk = poisoned[k] if (active and k in poisoned) else fd.y_train[ix]
            new_vec = local_train(work, xk, yk, cfg, gen, global_vec)
            updates[k] = (new_vec - global_vec)
            sizes.append(len(ix))
        sizes = np.array(sizes, dtype=np.float64)

        mal_here = [k for k in sel.tolist() if k in malicious]
        if active and cfg.attack in MODEL_POISON:
            updates = apply_model_poison(updates, mal_here, cfg.attack, gen,
                                         rho=cfg.attack_rho)
        if active and cfg.attack == "backdoor_boost":
            for k in mal_here:
                updates[k] = updates[k] * BACKDOOR_BOOST

        U = torch.stack([updates[k] for k in sel.tolist()])
        state["client_ids"] = sel.tolist()
        state["n_byz_est"] = max(1, int(round(cfg.byz_frac * len(sel))))

        if cfg.aggregator == "fltrust":
            vector_to_parameters(global_vec.clone(), work.parameters())
            rix = fd.root_idx
            state["root_update"] = local_train(work, fd.x_train[rix], fd.y_train[rix],
                                               cfg, gen, global_vec) - global_vec

        ta = time.time()
        delta, info = agg(U, sizes, state)
        agg_time += time.time() - ta
        global_vec = global_vec + delta

        lbl = [0 if (k in malicious and active) else 1 for k in sel.tolist()]
        rec = {"round": rnd, "clients": sel.tolist(), "labels": lbl,
               "w": np.asarray(info["w"]).tolist(), "sizes": sizes.tolist()}
        if "tau" in info:
            rec["tau"] = np.asarray(info["tau"]).tolist()
            rec["tau_inst"] = np.asarray(info["tau_inst"]).tolist()
            rec["hetero"] = float(info["hetero"])
            rec["span"] = np.asarray(info["span"]).tolist()
            if cfg.keep_log:
                rec["u"] = np.round(np.asarray(info["u"]), 5).tolist()
                th = _theory_record(info, U, delta, lbl, cfg.temperature)
                th["round"] = rnd
                theory.append(th)
        w_log.append(rec)

        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            vector_to_parameters(global_vec.clone(), model.parameters())
            acc, loss = evaluate(model, fd.x_test, fd.y_test)
            hist["round"].append(rnd)
            hist["acc"].append(acc)
            hist["loss"].append(loss)
            if x_bd is not None:
                hist["asr"].append(attack_success_rate(model, x_bd))
            if b_dir is not None:
                d = global_vec - theta0
                hist["proj"].append(float(torch.dot(d, b_dir)))
                hist["drift"].append(float(torch.linalg.norm(d)))
            if verbose:
                print(f"  r{rnd:3d} acc={acc:.4f} loss={loss:.4f}", flush=True)

    vector_to_parameters(global_vec.clone(), model.parameters())
    acc, loss = evaluate(model, fd.x_test, fd.y_test)
    detail = [evaluate_detailed(model, fd.x_test[ix], fd.y_test[ix], fd.num_classes)
              for ix in fd.client_test_idx]
    per_client = [d["acc"] for d in detail]
    per_client_bacc = [d["bacc"] for d in detail]
    per_client_f1 = [d["macro_f1"] for d in detail]
    ben = [k for k in range(len(detail)) if k not in malicious and np.isfinite(per_client[k])]
    benign_pc = [per_client[k] for k in ben]
    benign_bacc = [per_client_bacc[k] for k in ben]
    benign_f1 = [per_client_f1[k] for k in ben]

    # detection quality: assigned weight, benign vs malicious
    all_w, all_lab, all_tau = [], [], []
    for r in w_log:
        all_w.extend(r["w"])
        all_lab.extend(r["labels"])
        if "tau" in r:
            all_tau.extend(r["tau"])
    has_mal = any(l == 0 for l in all_lab)
    det_auc = auc_binary(all_w, all_lab) if has_mal else float("nan")
    tau_auc = auc_binary(all_tau, all_lab) if (has_mal and all_tau) else float("nan")
    mal_mass = float(np.mean([sum(w for w, l in zip(r["w"], r["labels"]) if l == 0)
                              for r in w_log])) if has_mal else 0.0
    # mass the sample-size prior alone would have given the adversaries
    prior_mass = float(np.mean([sum(s for s, l in zip(r["sizes"], r["labels"]) if l == 0)
                                / sum(r["sizes"]) for r in w_log])) if has_mal else 0.0

    out = {
        "config": asdict(cfg),
        "final_acc": acc,
        "final_loss": loss,
        "best_acc": max(hist["acc"]) if hist["acc"] else acc,
        "acc_last5": float(np.mean(hist["acc"][-5:])) if hist["acc"] else acc,
        "history": hist,
        "per_client_acc": per_client,
        "per_client_bacc": per_client_bacc,
        "per_client_macro_f1": per_client_f1,
        "client_sizes": [len(ix) for ix in fd.client_idx],
        "benign_jain": jain_index(benign_pc),
        "benign_worst10": worst_frac(benign_pc, 0.1),
        "benign_std": float(np.std(benign_pc)),
        "benign_mean": float(np.mean(benign_pc)),
        "benign_jain_bacc": jain_index(benign_bacc),
        "benign_worst10_bacc": worst_frac(benign_bacc, 0.1),
        "benign_mean_bacc": float(np.mean(benign_bacc)),
        "benign_jain_f1": jain_index(benign_f1),
        "benign_worst10_f1": worst_frac(benign_f1, 0.1),
        "benign_mean_f1": float(np.mean(benign_f1)),
        "det_auc": det_auc,
        "tau_auc": tau_auc,
        "mal_weight_mass": mal_mass,
        "prior_mass": prior_mass,
        "malicious": sorted(malicious),
        "num_params": num_params(model),
        "wall_time_s": time.time() - t0,
        "agg_time_s": agg_time,
        "weight_log": w_log if (cfg.keep_log or cfg.tag.startswith("keep")) else w_log[::5],
    }
    if x_bd is not None:
        out["asr_final"] = attack_success_rate(model, x_bd)
        out["asr_last5"] = float(np.mean(hist["asr"][-5:]))
    if theory:
        out["theory"] = theory
    return out
