# scripts/baselines/eval_pilot_baselines.py
# Purpose: quick pilot baselines vs Tube Rule A using only scenario-level CSV.

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict, List

GT_ALIASES   = ['success', 'ok_by_runs', 'sim_success', 'gt', 'label']
PRED_ALIASES = ['pred_ruleA', 'ok_by_tube', 'tube_pred', 'pred', 'ok_pred']

def first_existing(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

def coerce_binary(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.astype(int)
    sn = pd.to_numeric(s, errors='coerce')
    if sn.notna().all():
        return (sn > 0).astype(int)
    mapping_true  = {'true','t','yes','y','1','ok','pass','success'}
    mapping_false = {'false','f','no','n','0','fail','failed','failure','not ok','unsafe'}
    out = []
    for v in s.astype(str).str.strip().str.lower():
        if v in mapping_true:
            out.append(1)
        elif v in mapping_false:
            out.append(0)
        else:
            out.append(0)
    return pd.Series(out, index=s.index, dtype=int)

def confusion_counts(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str,int]:
    tp = int(((y_true==1) & (y_pred==1)).sum())
    tn = int(((y_true==0) & (y_pred==0)).sum())
    fp = int(((y_true==0) & (y_pred==1)).sum())
    fn = int(((y_true==1) & (y_pred==0)).sum())
    return dict(tp=tp, tn=tn, fp=fp, fn=fn)

def compute_metrics(tp, tn, fp, fn) -> Dict[str,float]:
    total = tp+tn+fp+fn
    acc = (tp+tn)/total if total>0 else np.nan
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    bacc = 0.5*((tp/(tp+fn) if (tp+fn)>0 else 0.0) + (tn/(tn+fp) if (tn+fp)>0 else 0.0))
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, bacc=bacc, total=total)

def fmt_pct(x: float, digits=1) -> str:
    if np.isnan(x): return "–"
    return f"{100*x:.{digits}f}\\%"

def emit_latex_table(rows: List[Dict], outpath: Path, caption: str, label: str):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append(f"\\caption{{{caption}}}")
    tex.append(f"\\label{{{label}}}")
    tex.append("\\begin{tabular}{lcccc}")
    tex.append("\\hline")
    tex.append("Method & Accuracy & Precision & Recall & F1 \\\\ \\hline")
    for r in rows:
        tex.append(f"{r['method']} & {fmt_pct(r['accuracy'])} & {fmt_pct(r['precision'])} & "
                   f"{fmt_pct(r['recall'])} & {fmt_pct(r['f1'])} \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    outpath.write_text("\n".join(tex), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="Pilot baselines vs. Tube Rule A.")
    ap.add_argument("--s2t_csv", required=True, help="Path to scenario-level CSV.")
    ap.add_argument("--outdir", default="outputs/s2t_eval", help="Output directory.")
    ap.add_argument("--mu_thr", type=float, default=0.5, help="Feasible if mu >= mu_thr.")
    ap.add_argument("--v_thr", type=float, default=60.0, help="Feasible if v0 <= v_thr (km/h).")
    # kept for CLI compatibility
    ap.add_argument("--eta", type=float, default=0.95)
    ap.add_argument("--k", type=float, default=1.20)
    ap.add_argument("--emit_latex", action="store_true")
    args = ap.parse_args()

    csv_path = Path(args.s2t_csv)
    if not csv_path.exists():
        print("[ERR] CSV not found:", csv_path)
        return

    df = pd.read_csv(csv_path)
    col_gt   = first_existing(df.columns, GT_ALIASES)
    col_pred = first_existing(df.columns, PRED_ALIASES)
    if col_gt is None or col_pred is None:
        raise ValueError(f"CSV found but need columns: GT in {GT_ALIASES} and PRED in {PRED_ALIASES}. "
                         f"Found columns: {list(df.columns)}")
    print(f"[INFO] Using GT column:     {col_gt}")
    print(f"[INFO] Using Tube (A) col:  {col_pred}")

    y_true = coerce_binary(df[col_gt]).values
    y_tube = coerce_binary(df[col_pred]).values

    methods = {}
    methods["Tube (Rule A)"] = y_tube
    methods["Always-safe"] = np.ones_like(y_true)
    methods["Always-infeasible"] = np.zeros_like(y_true)
    majority = 1 if (y_true.mean() >= 0.5) else 0
    methods["Prevalence prior"] = np.full_like(y_true, fill_value=majority)

    if set(['mu','v0']).issubset(df.columns):
        mu = pd.to_numeric(df['mu'], errors='coerce').fillna(0.0).values
        v0 = pd.to_numeric(df['v0'], errors='coerce').fillna(0.0).values
        y_thr = ((mu >= args.mu_thr) & (v0 <= args.v_thr)).astype(int)
        methods[f"MuSpeed thr ($\\mu\\ge{args.mu_thr}$, $v_0\\le{int(args.v_thr)}$)"] = y_thr
    else:
        print("[WARN] 'mu'/'v0' columns not found; skip MuSpeed threshold baseline.")

    rows = []
    for name, y_pred in methods.items():
        conf = confusion_counts(y_true, y_pred)  # keys: tp/tn/fp/fn
        met  = compute_metrics(**conf)
        rows.append(dict(method=name, **met))

    print("===== Pilot Baselines vs Tube (Rule A) =====")
    for r in rows:
        print(f"{r['method']:<35} N={r['total']:>3} "
              f"Acc={r['accuracy']:.3f} Prec={r['precision']:.3f} Rec={r['recall']:.3f} F1={r['f1']:.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(outdir/"baselines_summary.csv", index=False, encoding="utf-8")

    if args.emit_latex:
        emit_latex_table(rows, outdir/"baselines_summary.tex",
                         "Pilot baselines vs.~Tube Rule~A (scenario-level).",
                         "tab:pilot_baselines")

if __name__ == "__main__":
    main()
