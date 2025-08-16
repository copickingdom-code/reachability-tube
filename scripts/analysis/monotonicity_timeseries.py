# scripts/analysis/monotonicity_timeseries.py
# Check friction monotonicity over time using the 4D per-slice volume files:
#   outputs/metrics/volume4d_timeseries_mu{mu}_v{v0}.csv
#
# It produces:
#   - outputs/metrics/monotonicity/monotonicity_v{v0}.png  (timeseries plot)
#   - outputs/metrics/monotonicity/monotonicity_summary.csv
#   - outputs/metrics/monotonicity/monotonicity_table.tex  (optional, with --emit_latex)
#
# Usage:
#   python scripts/analysis/monotonicity_timeseries.py --timeseries_dir outputs/metrics --v0 all --emit_latex
#
from __future__ import annotations
import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

plt.rcParams['figure.dpi'] = 150

def read_curve(timeseries_dir: Path, mu: float, v0: int) -> pd.DataFrame | None:
    """Load one timeseries CSV and return DataFrame with columns: time_index, t_s, volume4d."""
    f = timeseries_dir / f"volume4d_timeseries_mu{mu}_v{v0}.csv"
    if not f.exists():
        print(f"[WARN] Missing file: {f.as_posix()}")
        return None
    df = pd.read_csv(f)
    # Normalize column names if needed
    cols = {c.lower(): c for c in df.columns}
    # helpers
    def find(colnames, options):
        for c in options:
            if c in colnames: return c
        return None
    colnames = [c for c in df.columns]
    tcol = find([c.lower() for c in colnames], ['t_s','time_s','t','time'])
    icol = find([c.lower() for c in colnames], ['time_index','idx','i','k'])
    vcol = find([c.lower() for c in colnames], ['volume4d','vol4d','vol','volume'])
    if vcol is None:
        raise ValueError(f"{f.name}: cannot find 4D volume column (expect one of: volume4d/vol4d/vol/volume)")
    out = pd.DataFrame()
    if icol is not None: out['time_index'] = pd.to_numeric(df[df.columns[[c.lower() for c in colnames].index(icol)]], errors='coerce').astype('Int64')
    else: out['time_index'] = np.arange(len(df), dtype=int)
    if tcol is not None: out['t_s'] = pd.to_numeric(df[df.columns[[c.lower() for c in colnames].index(tcol)]], errors='coerce')
    else: out['t_s'] = out['time_index'] * 0.01  # fallback: 10 ms
    out['volume4d'] = pd.to_numeric(df[df.columns[[c.lower() for c in colnames].index(vcol)]], errors='coerce').astype(float)
    out['mu'] = float(mu); out['v0'] = int(v0)
    return out

def violations_pair(vol_low: np.ndarray, vol_high: np.ndarray, abs_tol=1e-6, rel_tol=1e-3):
    """Boolean mask where monotone inclusion (low<=high) is violated beyond tolerance."""
    tol = np.maximum(abs_tol, rel_tol * np.maximum.reduce([np.abs(vol_low), np.abs(vol_high)]))
    return vol_low > vol_high + tol

def ensure_dir(d: Path):
    d.mkdir(parents=True, exist_ok=True)

def plot_timeseries(dfs_by_mu: dict[float, pd.DataFrame], v0: int, outdir: Path, tol_abs=1e-6, tol_rel=1e-3):
    """Plot timeseries curves and annotate violation rates."""
    # align by time_index
    base_mu = sorted(dfs_by_mu.keys())
    merged = None
    for mu in base_mu:
        df = dfs_by_mu[mu][['time_index','t_s','volume4d']].copy()
        df = df.rename(columns={'volume4d': f'vol_{mu}'})
        merged = df if merged is None else pd.merge(merged, df[['time_index', f'vol_{mu}']], on='time_index', how='inner')
    if merged is None:
        return None, None
    t = merged['t_s'].values

    # violations
    chain_viols = None
    pair_stats = {}
    mus_sorted = sorted(base_mu)
    for (lo, hi) in zip(mus_sorted[:-1], mus_sorted[1:]):
        v_lo = merged[f'vol_{lo}'].values
        v_hi = merged[f'vol_{hi}'].values
        viol = violations_pair(v_lo, v_hi, abs_tol=tol_abs, rel_tol=tol_rel)
        frac = float(np.mean(viol)) if len(viol)>0 else np.nan
        pair_stats[(lo,hi)] = dict(frac=frac, count=int(viol.sum()), N=int(len(viol)))
    if len(mus_sorted) >= 3:
        v03 = merged[f'vol_{mus_sorted[0]}'].values
        v05 = merged[f'vol_{mus_sorted[1]}'].values
        v08 = merged[f'vol_{mus_sorted[2]}'].values
        viol_chain = (violations_pair(v03, v05, tol_abs, tol_rel) | violations_pair(v05, v08, tol_abs, tol_rel))
        chain_viols = dict(frac=float(np.mean(viol_chain)), count=int(viol_chain.sum()), N=int(len(viol_chain)))

    # plot
    ensure_dir(outdir)
    plt.figure(figsize=(6.0, 3.2))
    for mu in mus_sorted:
        plt.plot(t, merged[f'vol_{mu}'].values, label=f'μ={mu}')
    ann = []
    for (lo,hi), st in pair_stats.items():
        ann.append(f'viol(μ={lo}≤{hi})={st["frac"]*100:.2f}%')
    if chain_viols is not None:
        ann.append(f'viol(chain)={chain_viols["frac"]*100:.2f}%')
    plt.title(f'4D per-slice volume vs. time  (v0={v0} km/h)')
    plt.xlabel('time t (s)')
    plt.ylabel('4D slice volume  Vol_d(A_k)  (arb. units)')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', ncol=len(mus_sorted))
    plt.text(0.01, 0.02, " | ".join(ann), transform=plt.gca().transAxes, fontsize=8)
    png = outdir / f"monotonicity_v{v0}.png"
    plt.tight_layout()
    plt.savefig(png)
    plt.close()
    return pair_stats, chain_viols

def write_latex_table(summary_rows, out_tex: Path):
    lines = []
    lines.append(r"\begin{table}[!h]")
    lines.append(r"\centering")
    lines.append(r"\caption{Volumetric monotonicity check: fraction of time slices with violations (lower is better).}")
    lines.append(r"\label{tab:mu_mono}")
    lines.append(r"\begin{tabular}{lccc}")
    lines.append(r"\hline")
    lines.append(r"$v_0$ (km/h) & viol$(0.3\le 0.5)$ & viol$(0.5\le 0.8)$ & viol(chain) \\ \hline")
    for row in summary_rows:
        def pct(x):
            return f"{x*100:.2f}\%" if (x is not None and not math.isnan(x)) else "n/a"
        lines.append(f"{row['v0']} & {pct(row.get('frac_03_05'))} & {pct(row.get('frac_05_08'))} & {pct(row.get('frac_chain'))} \\\\")
    lines.append(r"\hline")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table}")
    out_tex.write_text("\n".join(lines), encoding='utf-8')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeseries_dir", type=str, default="outputs/metrics",
                    help="Directory containing volume4d_timeseries_mu*_v*.csv")
    ap.add_argument("--v0", type=str, default="all", help="Initial speed(s): all or comma list like 30,60,120")
    ap.add_argument("--mus", type=str, default="0.3,0.5,0.8", help="Friction levels to include (comma list)")
    ap.add_argument("--abs_tol", type=float, default=1e-6, help="Absolute tolerance for monotonicity (vol units)")
    ap.add_argument("--rel_tol", type=float, default=1e-3, help="Relative tolerance for monotonicity (fraction)")
    ap.add_argument("--emit_latex", action="store_true", help="Emit a LaTeX summary table")
    args = ap.parse_args()

    timeseries_dir = Path(args.timeseries_dir)
    outdir = timeseries_dir / "monotonicity"
    outdir.mkdir(parents=True, exist_ok=True)

    v0s = [30,60,120] if args.v0=="all" else [int(x) for x in args.v0.split(",")]
    mus = [float(x) for x in args.mus.split(",")]
    summary_rows = []
    for v0 in v0s:
        dfs_by_mu = {}
        for mu in mus:
            df = read_curve(timeseries_dir, mu, v0)
            if df is not None:
                dfs_by_mu[mu] = df
        if len(dfs_by_mu) < 2:
            print(f"[WARN] v0={v0}: need at least 2 μ curves; found {list(dfs_by_mu.keys())}. Skipping.")
            continue
        pair_stats, chain_stats = plot_timeseries(dfs_by_mu, v0, outdir, args.abs_tol, args.rel_tol)
        row = dict(v0=v0)
        if pair_stats is not None:
            for (lo,hi), st in pair_stats.items():
                if lo==0.3 and hi==0.5: row['frac_03_05']=st['frac']
                if lo==0.5 and hi==0.8: row['frac_05_08']=st['frac']
        if chain_stats is not None:
            row['frac_chain'] = chain_stats['frac']
        summary_rows.append(row)

    if summary_rows:
        sum_df = pd.DataFrame(summary_rows).sort_values('v0')
        sum_df.to_csv(outdir/"monotonicity_summary.csv", index=False, encoding='utf-8')
        print("Saved summary to", (outdir/"monotonicity_summary.csv").as_posix())
        if args.emit_latex:
            write_latex_table(summary_rows, outdir/"monotonicity_table.tex")
            print("Wrote LaTeX table to", (outdir/"monotonicity_table.tex").as_posix())
        print("Figures in", outdir.as_posix())
    else:
        print("[WARN] No summaries written; check inputs.")

if __name__ == "__main__":
    main()
