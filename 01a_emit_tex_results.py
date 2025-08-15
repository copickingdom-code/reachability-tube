# scripts/08a_emit_tex_results.py
# Robust TeX emitter for Paper A: acceptance table + key macros
import os, re, sys, glob
from pathlib import Path
import numpy as np
import pandas as pd

try:
    from vehicletube.paths import outputs_root
    from vehicletube.config import load as load_cfg
except Exception:
    # 回退：若包未就绪，则按当前仓库相对位置定位
    def outputs_root():
        return Path("outputs").resolve()
    def load_cfg(p):
        import yaml
        with open(p, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

# ---------- helpers ----------
def coalesce_col(df, *names):
    """返回第一个存在的列名，否则 None"""
    for n in names:
        if n in df.columns:
            return n
    return None

def percent_fmt(x, nd=1):
    if pd.isna(x):
        return "--"
    return f"{x*100:.{nd}f}\\%"

def sanitize_acceptance_table(p_csv):
    """读取并清洗 acceptance 表，返回标准列：['mu','v0_kph','acc_rate']"""
    df = pd.read_csv(p_csv)

    # 统一列名
    mu_col  = coalesce_col(df, "mu", "mu_val", "Mu")
    v0_col  = coalesce_col(df, "v0_kph", "v0", "speed_kph", "v0kmh", "V0")
    acc_col = coalesce_col(df, "acc_rate", "accept_rate", "acceptance", "acc")

    # 基本存在性检查
    if mu_col is None or v0_col is None:
        raise KeyError(f"Cannot find required columns for mu/v0 in {p_csv}. "
                       f"Have columns: {list(df.columns)}")

    # 若没有 acc_rate，就用 accepted/total 现算
    if acc_col is None:
        accepted_col = coalesce_col(df, "accepted", "stable", "num_accepted", "ok")
        total_col    = coalesce_col(df, "total", "num_total", "N", "all")
        if accepted_col is None or total_col is None:
            raise KeyError(f"Cannot find acceptance columns (acc_rate or accepted/total) in {p_csv}. "
                           f"Have columns: {list(df.columns)}")
        df["acc_rate"] = pd.to_numeric(df[accepted_col], errors="coerce") / \
                         pd.to_numeric(df[total_col],    errors="coerce")
        acc_col = "acc_rate"

    # 整型/浮点处理
    df = df.copy()
    df["mu"]      = pd.to_numeric(df[mu_col], errors="coerce")
    df["v0_kph"]  = pd.to_numeric(df[v0_col], errors="coerce")
    df["acc_rate"]= pd.to_numeric(df[acc_col], errors="coerce")
    df = df.dropna(subset=["mu","v0_kph","acc_rate"])

    # 排序 & 去重（若有重复组合，取均值）
    df = (df.groupby(["mu","v0_kph"], as_index=False)["acc_rate"]
            .mean()
            .sort_values(["mu","v0_kph"])
            .reset_index(drop=True))
    return df

def emit_acceptance_table_tex(df, out_tex):
    """把 df(mu,v0_kph,acc_rate) 透视为表格并写 TeX"""
    piv = df.pivot(index="v0_kph", columns="mu", values="acc_rate")
    v0_list = list(piv.index)
    mu_list = list(piv.columns)

    lines = []
    # 表头
    header_mu = " & ".join([f"$\\mu={m}$" for m in mu_list])
    lines.append("\\begin{table}[t]")
    lines.append("\\centering")
    lines.append("\\caption{Stable-trajectory acceptance rate (higher is better).}")
    lines.append("\\label{tab:acceptance}")
    lines.append("\\begin{tabular}{l" + "c"*len(mu_list) + "}")
    lines.append("\\hline")
    lines.append("v$_0$ (km/h) & " + header_mu + " \\\\ \\hline")
    # 表体
    for v0 in v0_list:
        row = [percent_fmt(piv.loc[v0, m]) for m in mu_list]
        lines.append(f"{int(v0)} & " + " & ".join(row) + " \\\\")
    lines.append("\\hline")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")

    out_tex.write_text("\n".join(lines), encoding="utf-8")

def emit_number_macros(df, out_tex):
    """输出一些常用宏：每个(mu,v0)的接受率 + 摩擦缩放拟合指数 p（若存在）"""
    lines = []
    lines.append("% Auto-generated; do not edit by hand.")
    # per (mu,v0)
    for mu, v0, acc in df[["mu","v0_kph","acc_rate"]].itertuples(index=False):
        mu_tag = str(mu).replace(".","p")
        lines.append(f"\\newcommand{{\\AccMu{mu_tag}V{int(v0)}kph}}{{{acc*100:.1f}\\%}}")
    # 拟合指数（若存在）
    fit_csv = outputs_root() / "metrics" / "mu_scaling_fit.csv"
    if fit_csv.exists():
        fit = pd.read_csv(fit_csv)
        p_col = coalesce_col(fit, "p", "exponent", "alpha")
        r2col = coalesce_col(fit, "r2", "R2", "r_squared")
        if p_col is not None:
            pval = float(fit[p_col].values[0])
            lines.append(f"\\newcommand{{\\MuScalingExponent}}{{{pval:.3f}}}")
        if r2col is not None:
            r2   = float(fit[r2col].values[0])
            lines.append(f"\\newcommand{{\\MuScalingRTwo}}{{{r2:.3f}}}")
    out_tex.write_text("\n".join(lines)+"\n", encoding="utf-8")

def main():
    out = outputs_root()
    tex_dir = out / "tex"
    tex_dir.mkdir(parents=True, exist_ok=True)

    # 1) 读取 + 清洗
    acc_csv = out / "metrics" / "acceptance_table.csv"
    if not acc_csv.exists():
        print(f"[ERR] {acc_csv} not found. Make sure 03_filter_stability.py / 05_eval_saturation.py ran.")
        sys.exit(1)

    df = sanitize_acceptance_table(acc_csv)
    # 保存清洗后的 CSV（便于排障与复用）
    df.to_csv(out / "metrics" / "acceptance_table_sanitized.csv", index=False, encoding="utf-8")

    # 2) 导出 TeX 表格
    emit_acceptance_table_tex(df, tex_dir / "acceptance_table.tex")
    # 3) 导出常用数字宏
    emit_number_macros(df, tex_dir / "results_numbers.tex")

    print(f"[OK] TeX emitted under {tex_dir}")

if __name__ == "__main__":
    main()
