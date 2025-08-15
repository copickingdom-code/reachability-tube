# scripts/07z_make_demo_scenarios.py
from pathlib import Path
import csv, re, random
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "scenarios"
OUT.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUT / "scenarios_minimal.csv"

# 想要每个 (mu, v0) 组合生成多少条
N_CUTIN = 8        # 右侧插入
N_PED   = 8        # 行人横穿
N_LEAD  = 8        # 前车减速/逼近

random.seed(2025)
np.random.seed(2025)

def discover_pairs():
    """扫描你已有的稳定清单，决定生成哪些 (mu, v0) 组合"""
    mani_dir = ROOT / "outputs" / "manifests"
    pairs = []
    for p in mani_dir.glob("stable_runs_mu*_v*.csv"):
        m = re.search(r"stable_runs_mu(.+)_v(\d+)\.csv", p.name)
        if not m: continue
        mu = m.group(1)     # 例如 "0.3"
        v0 = m.group(2)     # 例如 "60"
        pairs.append((mu, v0))
    pairs = sorted(set(pairs))
    if not pairs:
        # 兜底：用默认配置
        pairs = [("0.3","30"),("0.3","60"),("0.5","60"),("0.8","60"),("0.8","120")]
    return pairs

def clip(a, lo, hi): return max(lo, min(hi, a))

def gen_cutin(mu, v0, k):
    # 右侧相邻车道插入：初始在 y≈-3.7m（右车道中心），向左 vy>0
    x0 = np.random.uniform(8.0, 12.0)     # 前方距离
    y0 = np.random.normal(-3.8, 0.2)      # 右车道中心附近
    vx = np.random.uniform(-0.5, 1.5)     # 纵向相对速度：略慢/略快
    vy = np.random.uniform(1.8, 2.4)      # 左向插入
    L, W = 4.5, 1.8
    margin = 0.7                          # 安全裕度
    return dict(
        sid=f"cutin_{mu}_{v0}_{k:02d}", mu=mu, v0_kph=v0, kind="cutin",
        x0=f"{x0:.2f}", y0=f"{y0:.2f}", vx=f"{vx:.2f}", vy=f"{vy:.2f}",
        L_obs=L, W_obs=W, R_obs=0.0, margin_m=margin
    )

def gen_ped(mu, v0, k):
    # 行人自右向左横穿：起点在右侧路肩 y≈-2.5，vy>0
    x0 = np.random.uniform(8.0, 12.0)
    y0 = np.random.normal(-2.5, 0.4)
    vx = 0.0
    vy = np.random.uniform(1.8, 2.2)
    R  = 0.5
    margin = 0.6
    return dict(
        sid=f"ped_{mu}_{v0}_{k:02d}", mu=mu, v0_kph=v0, kind="ped_xing",
        x0=f"{x0:.2f}", y0=f"{y0:.2f}", vx=f"{vx:.2f}", vy=f"{vy:.2f}",
        L_obs=0.0, W_obs=0.0, R_obs=R, margin_m=margin
    )

def gen_lead(mu, v0, k):
    # 同车道前车：位于正前方 y≈0，给一个相对后退速度 vx<0（代表对向闭合/前车急减速）
    x0 = np.random.uniform(10.0, 16.0)
    y0 = np.random.normal(0.0, 0.1)
    vx = np.random.uniform(-8.0, -5.0)    # 向后（相对 ego）移动
    vy = 0.0
    L, W = 4.6, 1.9
    margin = 0.8
    return dict(
        sid=f"lead_{mu}_{v0}_{k:02d}", mu=mu, v0_kph=v0, kind="lead_brake",
        x0=f"{x0:.2f}", y0=f"{y0:.2f}", vx=f"{vx:.2f}", vy=f"{vy:.2f}",
        L_obs=L, W_obs=W, R_obs=0.0, margin_m=margin
    )

def main():
    pairs = discover_pairs()
    rows = []
    for mu, v0 in pairs:
        for k in range(N_CUTIN): rows.append(gen_cutin(mu, v0, k))
        for k in range(N_PED):   rows.append(gen_ped(mu, v0, k))
        for k in range(N_LEAD):  rows.append(gen_lead(mu, v0, k))

    fieldnames = ["sid","mu","v0_kph","kind","x0","y0","vx","vy",
                  "L_obs","W_obs","R_obs","margin_m"]
    with open(CSV_PATH, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows: w.writerow(r)

    print(f"[OK] wrote {CSV_PATH} with {len(rows)} scenarios across {len(pairs)} (mu,v0) pairs.")

if __name__ == "__main__":
    main()
