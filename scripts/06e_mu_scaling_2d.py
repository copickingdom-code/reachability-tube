# scripts/06e_mu_scaling.py  —— 2D 代理口径（xy union area）
# 读取 outputs/metrics/area_union_mu{mu}_v{v0}.csv
# 对时间积分得到 A2D = ∫ Area_union(t) dt，并按 v0 对 mu 做幂律拟合 A2D ≈ C * mu^p
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vehicletube.paths import outputs_root
from vehicletube.config import load

plt.rcParams['figure.dpi'] = 130

TIME_CANDIDATES = ['t_s','time_s','Time','time','t','Time (s)']
AREA2D_CANDIDATES = ['area_m2','union_area_m2','Area','area','A']

def find_col(cols, cands, fallback_last=False):
    for c in cands:
        if c in cols: return c
    return cols[-1] if fallback_last else None

def integrate_area(csv_path, dt_fallback):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)

    acol = find_col(cols, AREA2D_CANDIDATES, fallback_last=False)
    if acol is None:
        raise ValueError(f"No 2D area column found in {csv_path}. "
                         f"Try one of: {AREA2D_CANDIDATES}")

    tcol = find_col(cols, TIME_CANDIDATES, fallback_last=False)

    a = pd.to_numeric(df[acol], errors='coerce').astype(float).values
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if tcol is None:
        t = np.arange(len(a), dtype=float) * dt_fallback
    else:
        t = pd.to_numeric(df[tcol], errors='coerce').astype(float).values
        t = np.maximum.accumulate(np.nan_to_num(t, nan=0.0))

    return float(np.trapz(a, t))

def parse_mu_v0(path, pattern=r'area_union_mu([0-9.]+)_v(\d+)\.csv'):
    m = re.search(pattern, os.path.basename(path))
    if not m: return None, None
    return float(m.group(1)), int(m.group(2))

def fit_power(mu_arr, val_arr):
    mu_arr = np.asarray(mu_arr, float)
    val_arr = np.asarray(val_arr, float)
    mask = (mu_arr>0) & (val_arr>0)
    mu_arr, val_arr = mu_arr[mask], val_arr[mask]
    x, y = np.log(mu_arr), np.log(val_arr)
    p, logC = np.polyfit(x, y, 1)
    C = float(np.exp(logC))
    yhat = logC + p*x
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else np.nan
    return p, C, r2

def main():
    cfg = load("configs/default.yaml")
    out = outputs_root()
    dt = float(cfg.get('dt_s', 0.01))

    files = sorted(glob.glob((out/"metrics"/"area_union_mu*_v*.csv").as_posix()))
    if not files:
        print("[WARN] no area_union_mu*_v*.csv under outputs/metrics."); return

    rows = []
    for f in files:
        mu, v0 = parse_mu_v0(f)
        if mu is None:
            print("[SKIP] pattern not match:", f);
            continue
        try:
            A2D = integrate_area(f, dt_fallback=dt)
            rows.append(dict(mu=mu, v0=v0, A2D=A2D, file=f))
        except Exception as e:
            print("[ERR]", f, e)

    if not rows:
        print("[WARN] no valid rows"); return

    df = pd.DataFrame(rows).sort_values(['v0','mu']).reset_index(drop=True)
    (out/"metrics").mkdir(parents=True, exist_ok=True)
    (out/"figures").mkdir(parents=True, exist_ok=True)

    summary = []
    for v0, g in df.groupby('v0'):
        mu_arr  = g['mu'].values
        val_arr = g['A2D'].values
        if len(mu_arr) < 2:
            print(f"[WARN] v0={v0}: need at least 2 mus, got {len(mu_arr)}")
            continue

        p, C, r2 = fit_power(mu_arr, val_arr)
        print(f"[INFO] v0={v0}: ∫Area2D ≈ {C:.3g} * mu^{p:.3f}, R^2={r2:.3f}")

        g2 = g.copy()
        g2['fit_p'] = p; g2['fit_C'] = C; g2['R2'] = r2
        g2.to_csv(out/"metrics"/f"mu_scaling2d_v{v0}.csv", index=False, encoding="utf-8")

        xs = np.linspace(float(mu_arr.min())*0.9, float(mu_arr.max())*1.1, 100)
        xs[xs<=0] = np.min(mu_arr[mu_arr>0])
        ys = C * xs**p

        plt.figure()
        plt.loglog(mu_arr, val_arr, 'o', label='2D data (xy union)')
        plt.loglog(xs, ys, '-', label=f'fit: p={p:.3f}, R²={r2:.3f}')
        plt.xlabel('Friction μ')
        plt.ylabel('∫ Area₂ᴰ,union(t) dt  (m²·s)')
        plt.title(f'μ-scaling (2D proxy, v0 = {v0} km/h)')
        plt.legend()
        plt.tight_layout()
        figp = out/"figures"/f"mu_scaling2d_v{v0}.png"
        plt.savefig(figp); plt.close()

        summary.append(dict(v0=v0, p=p, C=C, R2=r2,
                            mu_min=float(mu_arr.min()), mu_max=float(mu_arr.max()),
                            n=len(mu_arr)))

    if summary:
        pd.DataFrame(summary).sort_values('v0').to_csv(out/"metrics"/"mu_scaling2d_summary.csv",
                                                       index=False, encoding="utf-8")
        print("Saved summary to", (out/"metrics"/"mu_scaling2d_summary.csv"))
        print("Figures in", (out/"figures"))

if __name__ == "__main__":
    main()
