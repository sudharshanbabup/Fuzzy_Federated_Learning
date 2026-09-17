"""Federated learning simulator."""
from __future__ import annotations

import copy
import time
from dataclasses import dataclass, asdict, field
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.utils import parameters_to_vector, vector_to_parameters

from .aggregators import AGGREGATORS, FedHIFT, build_aggregator
from .attacks import DATA_POISON, MODEL_POISON, apply_model_poison, poison_labels
from .data import FederatedData, load_federated
from .metrics import auc_binary, evaluate, jain_index, worst_frac
from .models import build_model, num_params


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
    eval_every: int = 5
    # FedHIFT hyper-parameters
    temperature: float = 0.20
    nu: float = 1.0
    beta_rep: float = 0.5
    sketch_dim: int = 16384
    type1: bool = False
    use_size_prior: bool = True
    criteria: List[str] = field(default_factory=lambda: ["align", "peer", "norm", "stab"])
    kappa: float | None = None
    tag: str = ""


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


def run(cfg: FLConfig, fd: FederatedData | None = None, verbose: bool = False) -> Dict:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.set_num_threads(1)
    rng = np.random.default_rng(cfg.seed + 7919)
    gen = torch.Generator().manual_seed(cfg.seed + 104729)

    if fd is None:
        fd = load_federated(cfg.dataset, cfg.num_clients, cfg.alpha, cfg.seed,
                            cfg.train_subsample, cfg.root_size)

    # The adversarial set is always drawn (even when it is unused) so that the
    # random stream that selects participants is identical across every
    # aggregation rule and attack for a given seed.
    n_byz = int(round(cfg.byz_frac * cfg.num_clients))
    drawn = set(rng.choice(cfg.num_clients, size=max(n_byz, 1), replace=False).tolist())
    malicious = drawn if (cfg.attack != "none" and n_byz > 0) else set()

    model = build_model(cfg.dataset)
    global_vec = parameters_to_vector(model.parameters()).detach().clone()
    work = build_model(cfg.dataset)

    # cached poisoned labels for data-poisoning adversaries
    poisoned: Dict[int, torch.Tensor] = {}
    if cfg.attack in DATA_POISON:
        for k in malicious:
            ix = fd.client_idx[k]
            poisoned[k] = poison_labels(fd.y_train[ix], cfg.attack, fd.num_classes, gen)

    if cfg.aggregator.startswith("fedhift"):
        from .fuzzy import IT2Config
        it2 = IT2Config()
        if cfg.kappa is not None:
            it2.kappa = cfg.kappa
        agg = FedHIFT(temperature=cfg.temperature, nu=cfg.nu, beta_rep=cfg.beta_rep,
                      sketch_dim=cfg.sketch_dim, type1=cfg.type1,
                      use_size_prior=cfg.use_size_prior, cfg=it2,
                      criteria=cfg.criteria, seed=cfg.seed)
    else:
        agg = AGGREGATORS[cfg.aggregator]

    state: Dict = {"n_byz_est": max(1, int(round(cfg.byz_frac * cfg.clients_per_round)))}
    hist = {"round": [], "acc": [], "loss": []}
    w_log: List[Dict] = []
    agg_time = 0.0
    t0 = time.time()

    for rnd in range(1, cfg.rounds + 1):
        sel = rng.choice(cfg.num_clients, size=cfg.clients_per_round, replace=False)
        updates: Dict[int, torch.Tensor] = {}
        sizes = []
        for k in sel:
            vector_to_parameters(global_vec.clone(), work.parameters())
            ix = fd.client_idx[k]
            xk = fd.x_train[ix]
            yk = poisoned[k] if k in poisoned else fd.y_train[ix]
            new_vec = local_train(work, xk, yk, cfg, gen, global_vec)
            updates[k] = (new_vec - global_vec)
            sizes.append(len(ix))
        sizes = np.array(sizes, dtype=np.float64)

        mal_here = [k for k in sel.tolist() if k in malicious]
        if cfg.attack in MODEL_POISON:
            updates = apply_model_poison(updates, mal_here, cfg.attack, gen)

        U = torch.stack([updates[k] for k in sel.tolist()])
        state["client_ids"] = sel.tolist()

        if cfg.aggregator == "fltrust":
            vector_to_parameters(global_vec.clone(), work.parameters())
            rix = fd.root_idx
            state["root_update"] = local_train(work, fd.x_train[rix], fd.y_train[rix],
                                               cfg, gen, global_vec) - global_vec

        ta = time.time()
        delta, info = agg(U, sizes, state)
        agg_time += time.time() - ta
        global_vec = global_vec + delta

        lbl = [0 if k in malicious else 1 for k in sel.tolist()]
        rec = {"round": rnd, "clients": sel.tolist(), "labels": lbl,
               "w": np.asarray(info["w"]).tolist()}
        if "tau" in info:
            rec["tau"] = np.asarray(info["tau"]).tolist()
            rec["hetero"] = float(info["hetero"])
            rec["span"] = np.asarray(info["span"]).tolist()
        w_log.append(rec)

        if rnd % cfg.eval_every == 0 or rnd == cfg.rounds:
            vector_to_parameters(global_vec.clone(), model.parameters())
            acc, loss = evaluate(model, fd.x_test, fd.y_test)
            hist["round"].append(rnd)
            hist["acc"].append(acc)
            hist["loss"].append(loss)
            if verbose:
                print(f"  r{rnd:3d} acc={acc:.4f} loss={loss:.4f}", flush=True)

    vector_to_parameters(global_vec.clone(), model.parameters())
    acc, loss = evaluate(model, fd.x_test, fd.y_test)
    per_client = [evaluate(model, fd.x_test[ix], fd.y_test[ix])[0]
                  for ix in fd.client_test_idx]
    benign_pc = [a for k, a in enumerate(per_client) if k not in malicious]

    # detection quality: mean assigned weight, benign vs malicious
    all_w, all_lab, all_tau = [], [], []
    for r in w_log:
        all_w.extend(r["w"])
        all_lab.extend(r["labels"])
        if "tau" in r:
            all_tau.extend(r["tau"])
    det_auc = auc_binary(all_w, all_lab) if malicious else float("nan")
    tau_auc = auc_binary(all_tau, all_lab) if (malicious and all_tau) else float("nan")
    mal_mass = float(np.mean([sum(w for w, l in zip(r["w"], r["labels"]) if l == 0)
                              for r in w_log])) if malicious else 0.0

    return {
        "config": asdict(cfg),
        "final_acc": acc,
        "final_loss": loss,
        "best_acc": max(hist["acc"]) if hist["acc"] else acc,
        "acc_last5": float(np.mean(hist["acc"][-5:])) if hist["acc"] else acc,
        "history": hist,
        "per_client_acc": per_client,
        "benign_jain": jain_index(benign_pc),
        "benign_worst10": worst_frac(benign_pc, 0.1),
        "benign_std": float(np.std(benign_pc)),
        "benign_mean": float(np.mean(benign_pc)),
        "det_auc": det_auc,
        "tau_auc": tau_auc,
        "mal_weight_mass": mal_mass,
        "malicious": sorted(malicious),
        "num_params": num_params(model),
        "wall_time_s": time.time() - t0,
        "agg_time_s": agg_time,
        "weight_log": w_log if cfg.tag.startswith("keep") else w_log[::5],
    }
