# scripts/closedloop/aeb_gate_eval.py
# Purpose: closed-loop style gate (AEB-like) evaluation on scenario-level CSV.
# Robust to column name aliases (GT: success/ok_by_runs; PRED: pred_ruleA/ok_by_tube)

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Dict

# ---------- Column aliases ----------
GT_ALIASES   = ['success', 'ok_by_runs', 'sim_success', 'gt', 'label']
PRED_ALIASES = ['pred_ruleA', 'ok_by_tube', 'tube_pred', 'pred', 'ok_pred']

def first_existing(cols, names):
    for n in names:
        if n in cols:
            return n
    return None

def coerce_binary(s: pd.Series) -> pd.Series:
    """Coerce series to {0,1} from numbers/strings/bools."""
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

def emit_overall_latex(conf: Dict[str,int], met: Dict[str,float], outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\caption{Geometric tube membership vs.~high-fidelity outcomes (Rule~A; overall).}")
    tex.append("\\label{tab:s2t_overall}")
    tex.append("\\begin{tabular}{lccc}")
    tex.append("\\hline")
    tex.append(" & Pred.~feasible & Pred.~infeasible & Total \\\\ \\hline")
    tex.append(f"Sim.~success & {conf['tp']} & {conf['fn']} & {conf['tp']+conf['fn']} \\\\")
    tex.append(f"Sim.~failure & {conf['fp']} & {conf['tn']} & {conf['fp']+conf['tn']} \\\\ \\hline")
    tex.append(f"Total & {conf['tp']+conf['fp']} & {conf['fn']+conf['tn']} & {met['total']} \\\\ \\hline")
    tex.append("\\end{tabular}")
    tex.append("")
    tex.append("% Metrics: accuracy, precision, recall, F1 (micro).")
    tex.append(f"\\vspace{{2pt}}\\small{{Accuracy = {fmt_pct(met['accuracy'])}, "
               f"Precision = {fmt_pct(met['precision'])}, "
               f"Recall = {fmt_pct(met['recall'])}, "
               f"F1 = {fmt_pct(met['f1'])}.}}")
    tex.append("\\end{table}")
    outpath.write_text("\n".join(tex), encoding="utf-8")

def emit_group_acc_latex(df: pd.DataFrame, y_true_col: str, y_pred_col: str, outpath: Path):
    outpath.parent.mkdir(parents=True, exist_ok=True)
    if not set(['mu','v0']).issubset(df.columns):
        return
    rows = []
    for (mu, v0), g in df.groupby(['mu','v0']):
        y = coerce_binary(g[y_true_col]).values
        p = coerce_binary(g[y_pred_col]).values
        acc = (y==p).mean() if len(g)>0 else np.nan
        rows.append((float(mu), int(v0), len(g), acc))
    rows.sort(key=lambda x: (x[0], x[1]))
    tex = []
    tex.append("\\begin{table}[t]")
    tex.append("\\centering")
    tex.append("\\caption{Extreme-scenario agreement by $(\\mu, v_0)$ group (Rule~A; accuracy).}")
    tex.append("\\label{tab:s2t_group}")
    tex.append("\\begin{tabular}{lccc}")
    tex.append("\\hline")
    tex.append("$\\mu$ & $v_0$ (km/h) & $N$ & Accuracy \\\\ \\hline")
    for mu, v0, n, acc in rows:
        tex.append(f"{mu:g} & {v0:d} & {n:d} & {fmt_pct(acc)} \\\\")
    tex.append("\\hline")
    tex.append("\\end{tabular}")
    tex.append("\\end{table}")
    outpath.write_text("\n".join(tex), encoding="utf-8")

def main():
    ap = argparse.ArgumentParser(description="AEB-like gate evaluation on S2T CSV.")
    ap.add_argument("--s2t_csv", required=True, help="Path to scenario-level CSV.")
    ap.add_argument("--outdir", default="outputs/s2t_eval", help="Output directory for summaries/LaTeX.")
    ap.add_argument("--emit_latex", action="store_true", help="Emit LaTeX tables.")
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

    print(f"[INFO] Using GT column:    {col_gt}")
    print(f"[INFO] Using PRED column:  {col_pred}")

    y_true = coerce_binary(df[col_gt]).values
    y_pred = coerce_binary(df[col_pred]).values

    conf = confusion_counts(y_true, y_pred)  # keys: tp/tn/fp/fn
    met  = compute_metrics(**conf)

    print("===== AEB Gate (Rule A) Summary =====")
    print(f"N={met['total']} | TP={conf['tp']} TN={conf['tn']} FP={conf['fp']} FN={conf['fn']}")
    print(f"Acc={met['accuracy']:.3f}  Prec={met['precision']:.3f}  Rec={met['recall']:.3f}  F1={met['f1']:.3f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Save CSV summary
    pd.DataFrame([{
        "N": met['total'], "TP": conf['tp'], "TN": conf['tn'], "FP": conf['fp'], "FN": conf['fn'],
        "accuracy": met['accuracy'], "precision": met['precision'], "recall": met['recall'],
        "f1": met['f1'], "balanced_accuracy": met['bacc']
    }]).to_csv(outdir/"aeb_gate_overall.csv", index=False, encoding="utf-8")

    # Group breakdown if mu,v0 available
    if set(['mu','v0']).issubset(df.columns):
        grp = []
        for (mu, v0), g in df.groupby(['mu','v0']):
            y = coerce_binary(g[col_gt]).values
            p = coerce_binary(g[col_pred]).values
            acc = (y == p).mean()
            grp.append(dict(mu=float(mu), v0=int(v0), N=len(g), accuracy=acc))
        pd.DataFrame(grp).sort_values(['mu','v0']).to_csv(outdir/"aeb_gate_group.csv", index=False, encoding="utf-8")

    if args.emit_latex:
        emit_overall_latex(conf, met, outdir/"aeb_gate_overall.tex")
        emit_group_acc_latex(df, col_gt, col_pred, outdir/"aeb_gate_group.tex")
        print("[INFO] LaTeX tables written to:", outdir)

if __name__ == "__main__":
    main()
