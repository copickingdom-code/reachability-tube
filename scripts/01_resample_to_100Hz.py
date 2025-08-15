# scripts/01_resample_to_100Hz.py
# 将 data/raw_runs/{mu}/{v0}/run_*.csv 从 ~1 ms 采样统一为 10 ms (100 Hz)
# 输出到 data/raw_runs_100Hz 保留原目录结构与文件名

import os, sys, math, glob
import numpy as np
import pandas as pd

IN_ROOT  = os.path.join("data", "raw_runs")
OUT_ROOT = os.path.join("data", "raw_runs_100Hz")
os.makedirs(OUT_ROOT, exist_ok=True)

TARGET_DT = 0.01      # 10 ms
WINDOW    = 2.0       # 2 s → 200 帧
ANGLE_COLS = ["Yaw","Yaw (deg)"]  # 若列名不同再加

def find_time_col(cols):
    cand = ["Time", "T_Stamp", "time", "t", "Time (s)"]
    for c in cand:
        if c in cols: return c
    return None

def unwrap_deg(a_deg):
    # 角度解缠，避免 179°→-179° 插值跳变
    rad = np.deg2rad(a_deg)
    rad_u = np.unwrap(rad)
    return np.rad2deg(rad_u)

def resample_frame(df):
    cols = list(df.columns)
    tcol = find_time_col(cols)
    if tcol is None:
        # 没有时间列就假定 1 ms 递增
        dt_est = 0.001
        t0 = 0.0
        t_in = t0 + np.arange(len(df))*dt_est
    else:
        t_in = pd.to_numeric(df[tcol], errors="coerce").values.astype(float)
        # 对偶发 NaN 做一次线性填充
        mask = np.isfinite(t_in)
        if not np.all(mask):
            t_in = np.interp(np.arange(len(t_in)), np.where(mask)[0], t_in[mask])
        t0 = t_in[0]
        # 对某些文件不是从 0 开始无所谓，统一以首样本为起点
    # 目标 200 帧（0 … 1.99s）
    t_grid = (t0 + np.arange(0.0, WINDOW, TARGET_DT)).astype(float)

    # 估计输入采样间隔（取中位数）
    if len(t_in) > 1:
        dt_in = np.median(np.diff(t_in))
    else:
        dt_in = 0.001

    # --- 优先走“等间隔抽点”路径（快）---
    use_decimate = (abs(dt_in - 0.001) < 2e-4) and (len(t_in) >= int(WINDOW/0.001))
    df_out = pd.DataFrame(index=np.arange(len(t_grid)))
    df_out["Time"] = t_grid

    if use_decimate:
        # 计算索引：将目标时间换算到输入索引并四舍五入
        idx = np.clip(np.round((t_grid - t0)/dt_in).astype(int), 0, len(df)-1)
        picked = df.iloc[idx].reset_index(drop=True).copy()
        # 处理角度列（抽点无需解缠，但保持一致性）
        for c in ANGLE_COLS:
            if c in picked.columns:
                picked[c] = picked[c].astype(float).values
        picked["Time"] = t_grid
        return picked
    else:
        # --- 稳健路径：对所有数值列做线性重采样 ---
        for c in df.columns:
            if c == tcol:
                continue
            # 仅对数值列插值
            s = pd.to_numeric(df[c], errors="coerce").values.astype(float)
            # 角度列先解缠再插值（避免跨 ±180°）
            if c in ANGLE_COLS:
                s = unwrap_deg(s)
            # 若输入时间非严格单调，先做单调化（微小扰动）
            t_fix = np.maximum.accumulate(t_in + 1e-12*np.arange(len(t_in)))
            # 插值：对 NaN 用边界外推
            valid = np.isfinite(s)
            if valid.sum() < 2:
                y = np.full_like(t_grid, np.nan, dtype=float)
            else:
                y = np.interp(t_grid, t_fix[valid], s[valid])
            df_out[c] = y
        return df_out

def process_one(in_path):
    rel = os.path.relpath(in_path, IN_ROOT)
    out_path = os.path.join(OUT_ROOT, rel).replace("\\","/")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    df = pd.read_csv(in_path)
    df_ds = resample_frame(df)

    # 保证恰好 200 行（0..1.99s）
    if len(df_ds) > int(WINDOW/TARGET_DT):
        df_ds = df_ds.iloc[:int(WINDOW/TARGET_DT)]
    elif len(df_ds) < int(WINDOW/TARGET_DT):
        # 末尾补最后一行
        last = df_ds.iloc[[-1]].copy()
        need = int(WINDOW/TARGET_DT) - len(df_ds)
        df_ds = pd.concat([df_ds, pd.concat([last]*need, ignore_index=True)], ignore_index=True)

    df_ds.to_csv(out_path, index=False, encoding="utf-8")
    return out_path

def main():
    files = sorted(glob.glob(os.path.join(IN_ROOT, "*", "*", "run_*.csv")))
    if not files:
        print(f"[WARN] no files under {IN_ROOT}/{{mu}}/{{v0}}/run_*.csv")
        sys.exit(0)
    print(f"[INFO] found {len(files)} files")
    ok = 0
    for k, f in enumerate(files, 1):
        try:
            out = process_one(f)
            ok += 1
            if k % 50 == 0 or k <= 5:
                print(f"[{k}/{len(files)}] → {out}")
        except Exception as e:
            print(f"[ERR] {f}: {e}")
    print(f"[DONE] {ok}/{len(files)} files → {OUT_ROOT}")

if __name__ == "__main__":
    main()
