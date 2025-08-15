
import os, glob, yaml, json, numpy as np, pandas as pd
from vehicletube.config import load
from vehicletube.paths import data_root, outputs_root
from vehicletube.io import read_run_csv
from vehicletube.stability import stability_filter

def merge_vehicle(cfg):
    par_yaml = 'configs/vehicle_from_par.yaml'
    if os.path.exists(par_yaml):
        with open(par_yaml,'r',encoding='utf-8') as f:
            y = yaml.safe_load(f) or {}
            if 'vehicle' in y: cfg['vehicle'] = y['vehicle']
    return cfg

def main():
    cfg = load('configs/default.yaml'); cfg = merge_vehicle(cfg)
    # >>> 加上这几行，确保是数值 <<<
    cfg['mus'] = [float(x) for x in cfg['mus']]
    cfg['v0_kph'] = [int(x) for x in cfg['v0_kph']]

    out_dir = outputs_root(); (out_dir/'manifests').mkdir(exist_ok=True, parents=True)
    (out_dir/'metrics').mkdir(exist_ok=True, parents=True); (out_dir/'figures').mkdir(exist_ok=True, parents=True)
    root = data_root()/ 'raw_runs_100Hz'
    summary = []
    for mu in cfg['mus']:
        for v0 in cfg['v0_kph']:
            files = sorted(glob.glob(os.path.join(root.as_posix(), f'{mu}', f'{v0}', 'run_*.csv')))
            if not files: print('[WARN] no files for', mu, v0); continue
            acc, tot = 0, 0
            fails = dict(S1_beta=0,S2_r=0,S3_ltr=0,S4_ax=0,S5_spec=0)
            stable = []
            for f in files:
                tot += 1
                tr = read_run_csv(f, horizon_s=cfg['horizon_s'], dt=cfg['dt_s'])
                res = stability_filter(tr, cfg, cfg['vehicle'], mu)
                if res['ok']:
                    acc += 1; stable.append(f)
                else:
                    for k in ['S1_beta','S2_r','S3_ltr','S4_ax','S5_spec']:
                        if not res['reasons'].get(k, True): fails[k]+=1; break
            pd.DataFrame(dict(run=stable)).to_csv(out_dir/'manifests'/f'stable_runs_mu{mu}_v{v0}.csv', index=False)
            summary.append(dict(mu=mu, v0=v0, total=tot, accepted=acc, acc_rate=acc/max(tot,1), **fails))
            print(f'[mu={mu} v0={v0}] {acc}/{tot} accepted')
    pd.DataFrame(summary).to_csv(out_dir/'metrics'/'acceptance_table.csv', index=False)

if __name__=='__main__':
    main()
