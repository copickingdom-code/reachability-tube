
import os, glob, json, numpy as np, pandas as pd, yaml
from vehicletube.config import load
from vehicletube.paths import data_root, outputs_root
from vehicletube.io import read_run_csv
from vehicletube.geometry_bins import build_layered_tube
from vehicletube.metrics import layered_areas_to_frame, union_area_per_time

def merge_vehicle(cfg):
    par_yaml = 'configs/vehicle_from_par.yaml'
    if os.path.exists(par_yaml):
        with open(par_yaml,'r',encoding='utf-8') as f:
            y = yaml.safe_load(f) or {}
            if 'vehicle' in y: cfg['vehicle'] = y['vehicle']
    return cfg

def main():
    cfg = load('configs/default.yaml'); cfg = merge_vehicle(cfg)
    out_dir = outputs_root(); (out_dir/'manifests').mkdir(exist_ok=True, parents=True)
    (out_dir/'metrics').mkdir(exist_ok=True, parents=True); (out_dir/'figures').mkdir(exist_ok=True, parents=True)
    root = data_root()/ 'raw_runs_100Hz'
    for mu in cfg['mus']:
        for v0 in cfg['v0_kph']:
            stab_csv = out_dir/'manifests'/f'stable_runs_mu{mu}_v{v0}.csv'
            if not stab_csv.exists(): print('[WARN] no stable list for', mu, v0); continue
            files = pd.read_csv(stab_csv)['run'].tolist()
            if not files: print('[WARN] empty stable runs for', mu, v0); continue
            K = int(cfg['horizon_s']/cfg['dt_s'])
            slices = [dict(x=[],y=[],v=[],psi=[]) for _ in range(K)]
            for f in files:
                tr = read_run_csv(f, horizon_s=cfg['horizon_s'], dt=cfg['dt_s'])
                for k in range(min(K, len(tr['t']))):
                    slices[k]['x'].append(tr['x'][k]); slices[k]['y'].append(tr['y'][k])
                    slices[k]['v'].append(tr['v_mps'][k]); slices[k]['psi'].append(tr['yaw_rad'][k])
            layered = build_layered_tube(slices, cfg['bins']['v_mps'], cfg['bins']['psi_rad'], cfg['alpha'])
            (out_dir/'manifests'/f'tube_layered_{mu}_{v0}.json').write_text(json.dumps(dict(mu=mu,v0=v0,dt=cfg['dt_s'],horizon=cfg['horizon_s'],layered=layered)), encoding='utf-8')
            df = layered_areas_to_frame(layered); df.to_csv(out_dir/'metrics'/f'areas_layered_mu{mu}_v{v0}.csv', index=False)
            union = union_area_per_time(layered); union.to_csv(out_dir/'metrics'/f'area_union_mu{mu}_v{v0}.csv', index=False)
            print(f'[mu={mu} v0={v0}] layered tube done')
if __name__=='__main__':
    main()
