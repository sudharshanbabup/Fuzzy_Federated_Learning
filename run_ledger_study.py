"""Ledger study of the OJCS revision: 80 runs, 5 long horizon reruns with full
weight logs and 6 budget sweep runs under the boosted backdoor. Usage: python run_ledger_study.py [workers]
Each run is stored as results_ledger/<name>.json."""
import itertools, json, os, sys, time
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
RES = os.path.join(HERE, "results_ledger")
os.makedirs(RES, exist_ok=True)
J = []
def add(name, cfg): J.append((name, cfg))
FM=dict(dataset="fmnist",num_clients=30,clients_per_round=10,rounds=30,local_epochs=1,batch_size=32,lr=0.05,momentum=0.9,train_subsample=20000,eval_every=3)
LH={**FM,"rounds":200,"train_subsample":10000,"eval_every":10,"track_dir":True}
for agg,c in [("fedavg_clip",None),("fedhift",None),("fedhift",1.25)]:
    tag=agg+("_L" if c else "")
    add(f"lh_{tag}_none",{**LH,"aggregator":agg,"attack":"none","seed":0,"alpha":0.5,"byz_frac":0.2,"ledger_c":c})
    for rho in [0.0,0.7]:
        add(f"lh_{tag}_ad{rho}",{**LH,"aggregator":agg,"attack":"adaptive","attack_rho":rho,"seed":0,"alpha":0.5,"byz_frac":0.2,"ledger_c":c})
for c in [1.1,1.5,2.0]:
    add(f"lh_fedhift_c{c}_ad0.0",{**LH,"aggregator":"fedhift","attack":"adaptive","attack_rho":0.0,"seed":0,"alpha":0.5,"byz_frac":0.2,"ledger_c":c})
ATK8=["none","label_flip","sign_flip","gauss","scaling","alie","ipm","adaptive"]
for s,a,(agg,c) in itertools.product([0,1,2],ATK8,[("fedhift",None),("fedhift",1.25)]):
    add(f"main_{agg}{'_L' if c else ''}_{a}_s{s}",{**FM,"aggregator":agg,"attack":a,"seed":s,"alpha":0.5,"byz_frac":0.2,"ledger_c":c})
for s,al,(agg,c) in itertools.product([0,1,2],[0.05,0.1],[("fedhift",None),("fedhift",1.25)]):
    add(f"nid_{agg}{'_L' if c else ''}_a{al}_s{s}",{**FM,"aggregator":agg,"attack":"label_flip","seed":s,"alpha":al,"byz_frac":0.2,"ledger_c":c})
for s,(agg,c) in itertools.product([0,1],[("fedhift",None),("fedhift",1.25)]):
    add(f"bdb_{agg}{'_L' if c else ''}_s{s}",{**FM,"aggregator":agg,"attack":"backdoor_boost","seed":s,"alpha":0.5,"byz_frac":0.2,"ledger_c":c})
    add(f"onoff_{agg}{'_L' if c else ''}_s{s}",{**FM,"aggregator":agg,"attack":"sign_flip","attack_start":10,"seed":s,"alpha":0.5,"byz_frac":0.2,"ledger_c":c})
LH={**FM,"rounds":200,"train_subsample":10000,"eval_every":10,"track_dir":True,"keep_log":True}
for agg,c,att,rho in [("fedhift",1.25,"adaptive",0.0),("fedhift",None,"adaptive",0.0),("fedavg_clip",None,"adaptive",0.0),("fedhift",1.25,"adaptive",0.7),("fedhift",1.25,"none",0.0)]:
    tag=agg+("_L" if c else "")+("_ad%s"%rho if att!="none" else "_none")
    J.append(("lhk_"+tag,{**LH,"aggregator":agg,"attack":att,"attack_rho":rho,"seed":0,"alpha":0.5,"byz_frac":0.2,"ledger_c":c}))

# budget sweep under the boosted backdoor (Fig. 9c)
for c in [1.1, 1.5, 2.0]:
    for s in [0, 1]:
        J.append((f"bdb_fedhift_c{c}_s{s}", {**FM, "aggregator": "fedhift", "attack": "backdoor_boost", "seed": s, "alpha": 0.5,
                                            "byz_frac": 0.2, "ledger_c": c}))


def one(x):
    from fedhift.fl import FLConfig, run
    name, cfg = x
    out = os.path.join(RES, name + ".json")
    if os.path.exists(out):
        return
    t = time.time(); r = run(FLConfig(**cfg)); r["_time"] = time.time() - t; r["_cfg"] = cfg
    json.dump(r, open(out, "w"))
    print(name, round(r["_time"]), r.get("acc_last5"), flush=True)


if __name__ == "__main__":
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 2) as p:
        list(p.imap_unordered(one, J))
