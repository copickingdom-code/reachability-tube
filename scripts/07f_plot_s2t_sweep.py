# scripts/07f_plot_s2t_sweep.py
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, math
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

def col_get(df, keys):
    # 在列名里模糊匹配（小写）
    low = {c.lower(): c for c in df.columns}
    for k in keys:
        if k in low: return low[k]
    for k in keys:
        for c in df.columns:
            if k in c.lower(): return c
    raise KeyError(f"missing any of: {keys}")

def heatmap(ax, Z, xs, ys, vmin=0.0, vmax=1.0, title="", cmap="viridis"):
    im = ax.imshow(Z, origin="lower", aspect="auto", vmin=vmin, vmax=vmax, cmap=cmap)
    ax.set_xticks(range(len(xs))); ax.set_xticklabels([f"{x:g}" for x in xs], rotation=0)
    ax.set_yticks(range(len(ys))); ax.set_yticklabels([f"{y:g}" for y in ys])
    ax.set_xlabel("frac_thr"); ax.set_ylabel("area_eps (m^2)")
    ax.set_title(title)
    # 数值标注（可关：注释掉下段）
    for i in range(len(ys)):
        for j in range(len(xs)):
            ax.text(j, i, f"{Z[i,j]:.2f}", ha="center", va="center", fontsize=8, color="white" if Z[i,j]>0.5 else "black")
    return im

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/metrics/s2t_pr_sweep.csv")
    ap.add_argument("--outdir", default="outputs/figures/s2t")
    ap.add_argument("--recall-min", type=float, default=0.75)
    args = ap.parse_args()

    outdir = Path(args.outdir); outdir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv, encoding="utf-8-sig")

    # 标准化列名
    area_c = col_get(df, ["area","area_eps","area_eps_m2","area_m2"])
    frac_c = col_get(df, ["frac","frac_thr"])
    roi_c  = col_get(df, ["roi","roi_radius","roi_radius_m"])
    ero_c  = col_get(df, ["eros","erosion","erosion_m"])
    P_c    = col_get(df, ["p","prec","precision"])
    R_c    = col_get(df, ["r","rec","recall"])
    A_c    = col_get(df, ["a","acc","accuracy"])

    # 类型
    df[area_c]=df[area_c].astype(float)
    df[frac_c]=df[frac_c].astype(float)
    df[roi_c] =df[roi_c].astype(float)
    df[ero_c] =df[ero_c].astype(float)
    df[P_c]=df[P_c].astype(float); df[R_c]=df[R_c].astype(float); df[A_c]=df[A_c].astype(float)

    best_rows = []

    for (roi, eros), g in df.groupby([roi_c, ero_c]):
        g = g.copy().sort_values([area_c, frac_c])
        areas = sorted(g[area_c].unique())
        fracs = sorted(g[frac_c].unique())

        # --- 热力图（Recall） ---
        Z = np.zeros((len(areas), len(fracs)), float)
        for i,a in enumerate(areas):
            for j,f in enumerate(fracs):
                sub = g[(g[area_c]==a) & (g[frac_c]==f)]
                Z[i,j] = float(sub[R_c].iloc[0]) if len(sub) else np.nan

        fig, ax = plt.subplots(figsize=(6.2, 4.2))
        im = heatmap(ax, Z, fracs, areas, vmin=0, vmax=1, title=f"Recall heatmap (roi={roi:g}, eros={eros:g})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(outdir/f"s2t_heatmap_recall_roi{roi:g}_ero{eros:g}.png", dpi=240)
        plt.close(fig)

        # --- PR 散点 + 自动挑选 ---
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision"); ax.grid(True, ls=":", lw=0.7)
        ax.set_xlim(0.0, 1.02); ax.set_ylim(0.0, 1.02)
        ax.axvline(args.recall_min, ls="--", lw=1.0)

        cand = []
        for _,row in g.iterrows():
            R = float(row[R_c]); P = float(row[P_c])
            ax.plot(R, P, "o", ms=5, alpha=0.7)
            cand.append((R,P,row))
        # 筛选 Recall≥阈值、Precision 最大；并以 Accuracy 次序打破并列
        cand_df = g[g[R_c] >= args.recall_min].copy()
        if len(cand_df):
            cand_df = cand_df.sort_values([P_c, A_c], ascending=[False, False])
            best = cand_df.iloc[0]
            br, bp = float(best[R_c]), float(best[P_c])
            ax.plot(br, bp, "s", ms=9, mfc="none", mec="red", mew=2)
            ax.annotate(f"best  P={bp:.3f}, R={br:.3f}\narea={best[area_c]:g}, frac={best[frac_c]:g}",
                        xy=(br,bp), xytext=(min(br+0.03,0.9), min(bp+0.06,0.98)),
                        arrowprops=dict(arrowstyle="->", lw=1.0), fontsize=9)
            best_rows.append({
                "roi": roi, "erosion": eros,
                "area_eps_m2": float(best[area_c]),
                "frac_thr": float(best[frac_c]),
                "precision": bp, "recall": br, "accuracy": float(best[A_c])
            })
        fig.tight_layout()
        fig.savefig(outdir/f"s2t_pr_roi{roi:g}_ero{eros:g}.png", dpi=240)
        plt.close(fig)

    # 汇总并选全局最优（先 Recall>=阈值，再 Precision 最大，若并列按 Accuracy）
    out_best = Path("outputs/metrics")/"s2t_best_pick.csv"
    bd = pd.DataFrame(best_rows)
    if len(bd):
        bd.to_csv(out_best, index=False, encoding="utf-8")
        # 选全局最优
        gbest = bd.sort_values(["recall","precision","accuracy"], ascending=[False,False,False]).iloc[0].to_dict()
        print(f"[pick] global best (Recall≥{args.recall_min:.2f}): "
              f"roi={gbest['roi']:.2f}, eros={gbest['erosion']:.2f}, "
              f"area={gbest['area_eps_m2']:.3f}, frac={gbest['frac_thr']:.2f}, "
              f"P={gbest['precision']:.3f}, R={gbest['recall']:.3f}, A={gbest['accuracy']:.3f}")
        # 写一个可直接粘贴到 default.yaml 的片段
        snippet = (
            "s2t:\n"
            f"  area_eps_m2:  {gbest['area_eps_m2']:.3f}\n"
            f"  frac_thr:     {gbest['frac_thr']:.2f}\n"
            f"  roi_radius_m: {gbest['roi']:.2f}\n"
            "tube:\n"
            f"  erosion_m:    {gbest['erosion']:.2f}\n"
        )
        (Path("outputs/metrics")/"s2t_best_pick.txt").write_text(snippet, encoding="utf-8")
    else:
        print(f"[pick] no point reaches Recall≥{args.recall_min:.2f}. Try lowering --recall-min or sweeping a wider grid.")

if __name__ == "__main__":
    main()
