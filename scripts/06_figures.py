import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from vehicletube.paths import outputs_root
from vehicletube.config import load

plt.rcParams['figure.dpi'] = 130

# --------- helpers: robust column picking ---------
_TIME_CANDS = ['t_s','time_s','Time','time','Time (s)','time_index','k','T_Stamp']
_AREA_CANDS = ['area_m2','union_area_m2','area','Area','A']

def _pick_time_col(cols):
    for c in _TIME_CANDS:
        if c in cols: return c
    return None

def _pick_area_col(cols):
    for c in _AREA_CANDS:
        if c in cols: return c
    # fallback: 取除时间列之外的最后一列
    others = [c for c in cols if c not in _TIME_CANDS]
    return others[-1] if others else cols[-1]

def _find_area_union_path(out, mu, v0):
    # 兼容两种位置：outputs/metrics/… 和 outputs/metrics/tubes/…
    cands = [
        out/'metrics'/f'area_union_mu{mu}_v{v0}.csv',
        out/'metrics'/'tubes'/f'area_union_mu{mu}_v{v0}.csv',
    ]
    for p in cands:
        if p.exists(): return p
    return None

# --------- plotting ---------
def plot_acceptance(out):
    p = out/'metrics'/'acceptance_table.csv'
    if not p.exists(): return
    acc = pd.read_csv(p)
    if acc.empty: return
    if 'acc_rate' not in acc.columns and {'accepted','total'}.issubset(acc.columns):
        acc['acc_rate'] = acc['accepted'] / acc['total'].replace(0, np.nan)
    piv = acc.pivot(index='v0', columns='mu', values='acc_rate')
    ax = piv.plot(kind='bar')
    ax.set_xlabel('v0 (km/h)'); ax.set_ylabel('Acceptance rate')
    ax.set_title('Stable Trajectory Acceptance')
    plt.tight_layout(); plt.savefig(out/'figures'/'acceptance_bar.png'); plt.close()
    print('[OK] acceptance_bar.png')

def plot_area_vs_time(out, cfg):
    os.makedirs(out/'figures', exist_ok=True)
    dt_default = float(cfg.get('dt_s', 0.01))
    for mu in cfg['mus']:
        for v0 in cfg['v0_kph']:
            path = _find_area_union_path(out, mu, v0)
            if path is None:
                print(f'[SKIP] no area_union for mu={mu}, v0={v0}')
                continue
            df = pd.read_csv(path)
            cols = list(df.columns)
            tcol = _pick_time_col(cols)
            acol = _pick_area_col(cols)

            if tcol is None:
                t = np.arange(len(df), dtype=float) * dt_default
            else:
                t = pd.to_numeric(df[tcol], errors='coerce').values.astype(float)
                # 保证非递减，防止插值/绘图异常
                t = np.maximum.accumulate(np.nan_to_num(t, nan=0.0))

            area = pd.to_numeric(df[acol], errors='coerce').values.astype(float)
            area = np.nan_to_num(area, nan=0.0, posinf=0.0, neginf=0.0)

            plt.figure()
            plt.plot(t, area)
            plt.grid(True)
            plt.xlabel('time (s)')
            plt.ylabel('union area (m$^2$)')
            plt.title(f'Union slice area μ={mu}, v0={v0} km/h')
            outpng = out/'figures'/f'area_vs_time_mu{mu}_v{v0}.png'
            plt.tight_layout(); plt.savefig(outpng); plt.close()
            print(f'[OK] {outpng}')

def main():
    out = outputs_root()
    cfg = load('configs/default.yaml')
    # 类型统一（YAML 里可能是字符串）
    cfg['mus']    = [float(x) for x in cfg['mus']]
    cfg['v0_kph'] = [int(x)   for x in cfg['v0_kph']]

    plot_acceptance(out)
    plot_area_vs_time(out, cfg)
    print('Figures done.')

if __name__ == '__main__':
    main()
