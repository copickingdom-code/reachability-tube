# scripts/06e_mu_scaling.py  (4D版)
# 读取 outputs/metrics/volume4d_mu{mu}_v{v0}.csv
# 对每个 v0，基于 (mu, ∫Vol4D dt) 做幂律拟合  Vol4D ≈ C * mu^p

from pathlib import Path
import os, glob, math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def try_import_outputs_root():
    try:
        from vehicletube.paths import outputs_root
        return outputs_root()
    except Exception:
        return Path("outputs").resolve()

OUT = Path(try_import_outputs_root())

plt.rcParams['figure.dpi'] = 130

def fit_power(mu_arr, vol_arr):
    x = np.log(mu_arr); y = np.log(vol_arr)
    p, logC = np.polyfit(x, y, 1)
    C = float(np.exp(logC))
    yhat = logC + p*x
    ss_res = float(np.sum((y - yhat)**2))
    ss_tot = float(np.sum((y - np.mean(y))**2)) if len(y) > 1 else 0.0
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else float("nan")
    return p, C, r2

def main():
    met = OUT / "metrics"
    fig = OUT / "figures"
    fig.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob((met / "volume4d_mu*_v*.csv").as_posix()))
    if not files:
        print("[WARN] 没找到 volume4d_mu*_v*.csv；先运行 04b_volume4d_timeseries.py")
        return

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    # 兼容字段名
    if 'vol4d_time_integral' not in df.columns:
        raise ValueError("缺少 vol4d_time_integral 列")

    summary = []
    for v0, g in df.groupby('v0'):
        g = g.sort_values('mu')
        mu_arr  = g['mu'].to_numpy(dtype=float)
        vol_arr = g['vol4d_time_integral'].to_numpy(dtype=float)
        if len(mu_arr) < 2:
            print(f"[WARN] v0={v0}: 至少需要 2 个 μ 点")
            continue
        p, C, r2 = fit_power(mu_arr, vol_arr)
        print(f"[INFO] v0={v0}: Vol4D ≈ {C:.3g} * mu^{p:.4f}, R^2={r2:.4f}")

        # 存表
        g2 = g.copy()
        g2['fit_p'] = p; g2['fit_C'] = C; g2['R2'] = r2
        g2.to_csv(met / f"mu_scaling4d_v{int(v0)}.csv", index=False, encoding="utf-8")

        # 画图（log-log）
        xs = np.linspace(max(1e-3, mu_arr.min()*0.9), mu_arr.max()*1.1, 100)
        ys = C * xs**p
        plt.figure()
        plt.loglog(mu_arr, vol_arr, 'o', label='4D data')
        plt.loglog(xs, ys, '-', label=f'fit: p={p:.3f}, R²={r2:.4f}')
        plt.xlabel('Friction μ')
        plt.ylabel('∫ Volume4D(t) dt')
        plt.title(f'μ-scaling (4D, v0 = {int(v0)} km/h)')
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig / f"mu_scaling4d_v{int(v0)}.png")
        plt.close()

        summary.append(dict(v0=int(v0), p=p, C=C, R2=r2,
                            mu_min=float(mu_arr.min()), mu_max=float(mu_arr.max()),
                            n=len(mu_arr)))

    if summary:
        pd.DataFrame(summary).sort_values('v0').to_csv(
            met / "mu_scaling4d_summary.csv", index=False, encoding="utf-8")
        print("Saved mu_scaling4d_summary.csv")
        print("Figures in", fig)

if __name__ == "__main__":
    main()
