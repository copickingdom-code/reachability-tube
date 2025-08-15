# scripts/06e_mu_scaling.py
# 读取 outputs/metrics/area_union_mu{mu}_v{v0}.csv
# 对时间积分得到 Vol = ∫ Area(t) dt，并按每个 v0 对 mu 做幂律拟合 Vol ≈ C * mu^p
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]   # 仓库根目录
sys.path.insert(0, str(ROOT))                # 关键：让 import vehicletube 可用


import os, re, glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from vehicletube.paths import outputs_root
from vehicletube.config import load

plt.rcParams['figure.dpi'] = 130

def find_time_col(cols):
    cands = ['t_s','time_s','Time','time','t','Time (s)']
    for c in cands:
        if c in cols: return c
    return None

def find_area_col(cols):
    cands = ['area_m2','union_area_m2','Area','area','A']
    for c in cands:
        if c in cols: return c
    # 兜底：取最后一列
    return cols[-1]

def integrate_area(csv_path, dt_fallback):
    df = pd.read_csv(csv_path)
    cols = list(df.columns)
    acol = find_area_col(cols)
    tcol = find_time_col(cols)

    a = pd.to_numeric(df[acol], errors='coerce').values.astype(float)
    a = np.nan_to_num(a, nan=0.0, posinf=0.0, neginf=0.0)

    if tcol is None:
        # 没有时间列，用配置里的 dt 和行数构造
        t = np.arange(len(a), dtype=float) * dt_fallback
    else:
        t = pd.to_numeric(df[tcol], errors='coerce').values.astype(float)
        # 若 t 不是严格单调，做一次单调化
        t = np.maximum.accumulate(np.nan_to_num(t, nan=0.0))

    # 用梯形法积分
    vol = float(np.trapz(a, t))
    return vol

def parse_mu_v0(path):
    # 匹配 area_union_mu0.5_v60.csv 这种
    m = re.search(r'area_union_mu([0-9.]+)_v(\d+)\.csv', os.path.basename(path))
    if not m: return None, None
    mu = float(m.group(1)); v0 = int(m.group(2))
    return mu, v0

def fit_power(mu_arr, vol_arr):
    # 在 log-log 空间线性回归: log V = log C + p log mu
    x = np.log(mu_arr); y = np.log(vol_arr)
    p, logC = np.polyfit(x, y, 1)
    C = float(np.exp(logC))
    # R^2
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
        print("[WARN] no area_union_mu*_v*.csv under outputs/metrics. Run 04/06 first.")
        return

    # 收集 vol(mu, v0)
    rows = []
    for f in files:
        mu, v0 = parse_mu_v0(f)
        if mu is None:
            print("[SKIP] pattern not match:", f);
            continue
        try:
            vol = integrate_area(f, dt_fallback=dt)
            rows.append(dict(mu=mu, v0=v0, vol=vol, file=f))
        except Exception as e:
            print("[ERR]", f, e)

    if not rows:
        print("[WARN] no valid rows"); return

    df = pd.DataFrame(rows).sort_values(['v0','mu']).reset_index(drop=True)
    (out/"metrics").mkdir(parents=True, exist_ok=True)
    (out/"figures").mkdir(parents=True, exist_ok=True)

    # 按 v0 分组拟合，并画图
    summary = []
    for v0, g in df.groupby('v0'):
        mu_arr  = g['mu'].values
        vol_arr = g['vol'].values
        if len(mu_arr) < 2:
            print(f"[WARN] v0={v0}: need at least 2 mus, got {len(mu_arr)}")
            continue

        p, C, r2 = fit_power(mu_arr, vol_arr)
        print(f"[INFO] v0={v0}: Vol ≈ {C:.3g} * mu^{p:.3f}, R^2={r2:.3f}")

        # 存表
        g2 = g.copy()
        g2['fit_p'] = p; g2['fit_C'] = C; g2['R2'] = r2
        g2.to_csv(out/"metrics"/f"mu_scaling_v{v0}.csv", index=False, encoding="utf-8")

        # 画图（log-log）
        mumin, mumax = float(mu_arr.min()), float(mu_arr.max())
        xs = np.linspace(max(1e-3,mumin*0.9), mumax*1.1, 100)
        ys = C * xs**p

        plt.figure()
        plt.loglog(mu_arr, vol_arr, 'o', label='data')
        plt.loglog(xs, ys, '-', label=f'fit: p={p:.3f}, R²={r2:.3f}')
        plt.xlabel('Friction μ')
        plt.ylabel('∫ Area_union(t) dt  (m²·s)')
        plt.title(f'μ-scaling (v0 = {v0} km/h)')
        plt.legend()
        plt.tight_layout()
        figp = out/"figures"/f"mu_scaling_v{v0}.png"
        plt.savefig(figp); plt.close()

        summary.append(dict(v0=v0, p=p, C=C, R2=r2,
                            mu_min=mumin, mu_max=mumax,
                            n=len(mu_arr)))

    # 汇总表
    if summary:
        pd.DataFrame(summary).sort_values('v0').to_csv(out/"metrics"/"mu_scaling_summary.csv",
                                                       index=False, encoding="utf-8")
        print("Saved summary to", (out/"metrics"/"mu_scaling_summary.csv"))
        print("Figures in", (out/"figures"))

if __name__ == "__main__":
    main()
