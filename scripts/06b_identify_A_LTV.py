# scripts/06b_identify_A_LTV.py
# 从稳定通过的 run 中，数据驱动识别 (beta, r) 的局部线性 A 矩阵，画出谱实部（稳定裕度）对比。
import os, glob, numpy as np, pandas as pd, matplotlib.pyplot as plt
from vehicletube.paths import data_root, outputs_root
from vehicletube.config import load
from vehicletube.io import read_run_csv

plt.rcParams['figure.dpi'] = 300

def identify_A_beta_r(tr, v_min=0.5):
    """最小二乘：X_{k+1}-X_k = dt * A * X_k  =>  A = (Y X^T) (dt X X^T)^(-1)"""
    beta = tr['beta_rad']; r = tr['r_rps']; t = tr['t']; v = tr['v_mps']
    if beta is None or r is None or t is None or len(beta) < 5:
        return None
    # 只用速度足够的样本，避免 0 速噪声
    mask = v >= v_min
    if mask.sum() < 5:
        return None
    beta = beta[mask]; r = r[mask]; t = t[mask]

    # 均匀步长近似
    dt = float(np.median(np.diff(t)))
    if dt <= 0 or not np.isfinite(dt):
        return None

    Xk  = np.vstack([beta[:-1], r[:-1]])         # 2 x (N-1)
    Xk1 = np.vstack([beta[ 1:], r[ 1:]])         # 2 x (N-1)
    Y   = Xk1 - Xk                                # 2 x (N-1)

    XtX = (Xk @ Xk.T)                             # 2x2
    lam = 1e-6
    try:
        A = (Y @ Xk.T) @ np.linalg.inv(dt * XtX + lam*np.eye(2))   # 2x2
    except np.linalg.LinAlgError:
        return None
    return A

def run_one_slice(mu, v0, cfg, max_runs_plot=50):
    out = outputs_root()
    man = out / "manifests" / f"stable_runs_mu{mu}_v{v0}.csv"
    if not man.exists():
        print(f"[WARN] no stable list for mu={mu}, v0={v0}, skip")
        return

    files = pd.read_csv(man)['run'].tolist()
    if not files:
        print(f"[WARN] empty stable runs for mu={mu}, v0={v0}, skip")
        return

    lambdas, traces, dets, runs = [], [], [], []
    for f in files[:max_runs_plot]:
        try:
            tr = read_run_csv(f, horizon_s=cfg['horizon_s'], dt=cfg['dt_s'])
            A  = identify_A_beta_r(tr)
            if A is None:
                continue
            w, _ = np.linalg.eig(A)
            lambdas.append(np.max(np.real(w)))
            traces.append(np.trace(A))
            dets.append(np.linalg.det(A))
            runs.append(os.path.basename(f))
        except Exception as e:
            print(f"[ERR] {f}: {e}")

    if not lambdas:
        print(f"[WARN] no identified A for mu={mu}, v0={v0}")
        return

    # 保存 CSV
    df = pd.DataFrame(dict(run=runs, lambda_max=lambdas, trace=traces, det=dets))
    df.to_csv(out/"metrics"/f"spec_margin_mu{mu}_v{v0}.csv", index=False)

    # 画图
    plt.plot(lambdas, marker='o', linestyle='-')
    plt.axhline(0.0, color='k', linewidth=1)
    plt.xlabel("run index")
    plt.ylabel("max Re(λ)")
    plt.title(f"Identified spectral margin (β–r)  μ={mu}, v0={v0} km/h")
    plt.tight_layout()
    plt.savefig(out/"figures"/f"spec_margin_ident_mu{mu}_v{v0}.png")
    plt.close()

def main():
    cfg = load("configs/default.yaml")
    # 强制数值化（以防 YAML 字符串）
    cfg['mus']    = [float(x) for x in cfg['mus']]
    cfg['v0_kph'] = [int(x)   for x in cfg['v0_kph']]

    for mu in cfg['mus']:
        for v0 in cfg['v0_kph']:
            run_one_slice(mu, v0, cfg)

    print("Done identify. Figures in outputs/figures, CSV in outputs/metrics.")

if __name__ == "__main__":
    main()
