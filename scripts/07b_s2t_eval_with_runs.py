# -*- coding: utf-8 -*-
"""
07b — Scenario-to-Tube vs Stable Runs (Layer 2, robust manifest)
- 输入：
  - outputs/metrics/s2t_eval_geometry.csv           （07a 产物，含 ok-by-tube）
  - outputs/manifests/stable_runs_mu{mu}_v{v0}.csv  （稳定轨迹清单：可为 run_id 或 直接文件路径）
  - (轨迹)  outputs/ingested_100Hz/**/run_XXXX.csv 或 data/raw_runs_100Hz/**/run_XXXX.csv 等
  - configs/default.yaml（可选）：dt_s, horizon_s, vehicle: {L_m, W_m, ego_margin_m}, s2t: {roi_radius_m}
- 输出：
  - outputs/metrics/s2t_eval_with_runs.csv
  - outputs/metrics/s2t_confusion_matrix.csv
"""
from pathlib import Path
import sys, json, re
import numpy as np
import pandas as pd

from shapely.geometry import Polygon, Point
from shapely.affinity import rotate, translate

# ==== 路径/配置 ====
try:
    from vehicletube.paths import outputs_root
    from vehicletube.config import load as cfg_load
except Exception:
    def outputs_root() -> Path:
        return Path("outputs")
    def cfg_load(_path:str):
        return {}

# -------------- helpers --------------
def _log_header(title:str):
    print("="*8, title, "="*8)

def load_cfg():
    cfg = cfg_load("configs/default.yaml") or {}
    dt = float(cfg.get("dt_s", 0.01))
    T  = float(cfg.get("horizon_s", 2.0))
    veh= cfg.get("vehicle", {}) or {}
    L  = float(veh.get("L_m", 4.60))
    W  = float(veh.get("W_m", 1.85))
    ego_margin = float(veh.get("ego_margin_m", 0.20))
    s2t = cfg.get("s2t", {}) or {}
    roi_radius = float(s2t.get("roi_radius_m", 15.0))
    return dt, T, L, W, ego_margin, roi_radius

def find_stable_manifest(mu:str, v0:str) -> Path:
    p = outputs_root()/ "manifests"/ f"stable_runs_mu{mu}_v{v0}.csv"
    return p if p.exists() else None

def candidate_run_paths(mu:str, v0:str, run_id:int):
    """根据 run_id 推断多个常见位置"""
    rid = f"run_{int(run_id):04d}.csv"
    rel = Path(mu) / v0 / rid
    candidates = [
        Path("outputs/ingested_100Hz") / rel,
        Path("outputs/ingested") / rel,
        Path("data/raw_runs_100Hz") / rel,
        Path("data/raw_runs") / rel,
    ]
    return [p for p in candidates if p.exists()]

def pathify(v: str) -> Path:
    """将字符串转为 Path；相对路径尝试按常见根目录补全"""
    p = Path(str(v))
    if p.exists():
        return p
    # 尝试 outputs/ 前缀
    q = outputs_root() / p
    if q.exists():
        return q
    return p  # 兜底返回原始路径（后续 exists 再过滤）

def looks_like_path(s: str) -> bool:
    s = str(s)
    return (".csv" in s.lower()) or ("run_" in s.lower()) or ("\\" in s) or ("/" in s)

def extract_run_files_from_manifest(df_runs: pd.DataFrame, mu: str, v0: str):
    """
    尽量鲁棒地从 manifest 里拿到轨迹文件列表：
      1) 优先找“路径型列”（大多数值像路径/文件名）
      2) 如果没有，再找 ID 列（run_id/id/run/rid），提取数字并拼路径
    """
    # 1) 路径型列
    for col in df_runs.columns:
        vals = df_runs[col].dropna().astype(str)
        if len(vals) == 0:
            continue
        # 若超过 60% 的值看起来像路径，就当作路径列
        is_pathish = vals.map(looks_like_path)
        if is_pathish.mean() > 0.6:
            files = [pathify(v) for v in vals.tolist()]
            files = [p for p in files if p.exists()]
            if files:
                print(f"[07b] manifest uses path column: '{col}' (n={len(files)})")
                return files

    # 2) ID 列
    for col in ["run_id", "id", "run", "rid"]:
        if col in df_runs.columns:
            vals = df_runs[col].dropna().astype(str)
            ids = []
            for v in vals:
                m = re.search(r"(\d+)", v)  # 容忍 'run_0016' 这类字符串
                if m:
                    ids.append(int(m.group(1)))
            files = []
            for rid in ids:
                cands = candidate_run_paths(mu, v0, rid)
                if cands:
                    files.append(cands[0])
            if files:
                print(f"[07b] manifest uses id column: '{col}' (n={len(files)})")
                return files

    print("[07b] WARN: no usable column in manifest; 0 files extracted")
    return []

def oriented_box(x:float, y:float, yaw_deg:float, L:float, W:float):
    """以 (x,y) 为中心、朝向 yaw 的车辆矩形"""
    hw, hl = W/2.0, L/2.0
    poly = Polygon([(-hl,-hw),(+hl,-hw),(+hl,+hw),(-hl,+hw)])
    poly = rotate(poly, yaw_deg, origin=(0,0), use_radians=False)
    poly = translate(poly, xoff=x, yoff=y)
    return poly

def read_run_xyyaw(run_csv:Path):
    """稳健读取单次轨迹的 X/Y/Yaw/Time，返回 (t, x, y, yaw_deg)；任一缺失则返回 None"""
    try:
        df = pd.read_csv(run_csv)
    except Exception:
        return None

    # time
    t = None
    for c in ["Time","T_Stamp","time","t","Time (s)"]:
        if c in df.columns:
            t = pd.to_numeric(df[c], errors="coerce").values.astype(float)
            break
    if t is None:
        # 假设等间隔 0.01 s
        t = np.arange(len(df)) * 0.01

    # X
    x = None
    for c in ["X (m)","X","Xcg_SM","X_m","X_axis","X_axis (m)"]:
        if c in df.columns:
            x = pd.to_numeric(df[c], errors="coerce").values.astype(float); break
    # Y
    y = None
    for c in ["Y (m)","Y","Ycg_SM","Y_m","Y_axis","Y_axis (m)"]:
        if c in df.columns:
            y = pd.to_numeric(df[c], errors="coerce").values.astype(float); break
    # Yaw（deg）
    yaw = None
    for c in ["Yaw (deg)","Yaw_deg","Yaw","Heading (deg)"]:
        if c in df.columns:
            yaw = pd.to_numeric(df[c], errors="coerce").values.astype(float); break

    if x is None or y is None:
        return None
    if yaw is None:
        yaw = np.zeros_like(x)

    # 清 NaN
    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(yaw) & np.isfinite(t)
    if not m.any():
        return None
    return t[m], x[m], y[m], yaw[m]

def resample_indices(t_in:np.ndarray, t_grid:np.ndarray):
    """用最近邻把 t_in 对齐到 t_grid，返回索引"""
    if t_in.size == 0:
        return np.zeros_like(t_grid, dtype=int)
    idx = np.searchsorted(t_in, t_grid, side="left")
    idx = np.clip(idx, 0, len(t_in)-1)
    left = np.clip(idx-1, 0, len(t_in)-1)
    right= idx
    choose_left = (np.abs(t_in[left]-t_grid) <= np.abs(t_in[right]-t_grid))
    return np.where(choose_left, left, right)

def scenario_to_Ft(row:pd.Series, times:np.ndarray):
    """保持与 07a 相同的障碍运动学"""
    from shapely.geometry import Point
    kind = str(row.get("kind", "cutin"))
    x0 = float(row.get("x0", 12.0)); y0 = float(row.get("y0", -3.5))
    vx = float(row.get("vx",  0.5)); vy = float(row.get("vy",  1.5))
    L  = float(row.get("L_obs", 4.5)); W = float(row.get("W_obs", 1.8))
    R  = float(row.get("R_obs", 0.0))
    margin = float(row.get("margin_m", 0.6))
    # 构障碍几何
    if kind == "ped_xing":
        base = Point(0.0, 0.0).buffer(max(R, 0.4))
    else:
        hw, hl = W/2.0, L/2.0
        base = Polygon([(-hl,-hw),(+hl,-hw),(+hl,+hw),(-hl,+hw)])
    base = base.buffer(max(margin, 0.0))
    return [translate(base, xoff=x0 + vx*t, yoff=y0 + vy*t) for t in times]

def load_times_for(mu:str, v0:str, dt:float):
    p = outputs_root()/ "manifests"/ f"tube_layered_{mu}_{v0}.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    K = len(obj.get("layered", []))
    return np.arange(K, dtype=float)*dt

# -------------- main --------------
def main():
    dt, T, L, W, ego_margin, roi_radius = load_cfg()
    out = outputs_root()
    (out/"metrics").mkdir(parents=True, exist_ok=True)

    # 读取 07a 的几何结果（按 sid 合并）
    tube_csv = out/"metrics"/"s2t_eval_geometry.csv"
    print(f"[07b] read tube metrics: {tube_csv.resolve()}")
    if not tube_csv.exists():
        print("[ERR] missing s2t_eval_geometry.csv (run 07a first)"); sys.exit(1)
    df_tube = pd.read_csv(tube_csv, dtype={"sid": str}, encoding="utf-8-sig")

    # 读取场景（获取 mu,v0,sid 和运动学）
    scen_csv = Path("data/scenarios/scenarios_minimal.csv")
    print(f"[07b] read scenarios:   {scen_csv.resolve()}")
    rows = pd.read_csv(scen_csv, dtype={"sid": str}, encoding="utf-8-sig")

    print(f"[07b] params: dt={dt}s, T={T}s, veh L={L}m W={W}m, ego_margin={ego_margin}m, roi={roi_radius}m")

    results = []
    for idx, r in rows.iterrows():
        sid = str(r["sid"]); mu = str(r["mu"]); v0 = str(r["v0_kph"])

        # 读取 07a 判定（仅按 sid 合并）
        ok_by_tube = -1
        hit = df_tube[df_tube["sid"].astype(str) == sid]
        if len(hit) > 0:
            ok_by_tube = int(hit["ok"].values[0])

        # 找稳定轨迹清单
        mani = find_stable_manifest(mu, v0)
        if mani is None:
            results.append(dict(sid=sid, mu=mu, v0=v0, ok_by_runs=0, ok_by_tube=ok_by_tube))
            continue

        df_runs = pd.read_csv(mani, encoding="utf-8-sig")
        run_files = extract_run_files_from_manifest(df_runs, mu, v0)

        # 若没有轨迹文件，直接 0
        if not run_files:
            results.append(dict(sid=sid, mu=mu, v0=v0, ok_by_runs=0, ok_by_tube=ok_by_tube))
            continue

        # time grid & 障碍 Ft
        times = load_times_for(mu, v0, dt)
        if times is None or len(times)==0:
            results.append(dict(sid=sid, mu=mu, v0=v0, ok_by_runs=0, ok_by_tube=ok_by_tube))
            continue
        Ft = scenario_to_Ft(r, times)
        origin = Point(0.0,0.0)
        win_idx = [k for k in range(len(times)) if Ft[k].distance(origin) <= roi_radius]
        if not win_idx:
            results.append(dict(sid=sid, mu=mu, v0=v0, ok_by_runs=0, ok_by_tube=ok_by_tube))
            continue

        # 逐条稳定轨迹做全程碰撞检测（找到一条全程无碰就算 1）
        ok_by_runs = 0
        Lm = L + 2*ego_margin
        Wm = W + 2*ego_margin
        for run_csv in run_files:
            arr = read_run_xyyaw(run_csv)
            if arr is None:
                continue
            t_in, x_in, y_in, yaw_in = arr
            # 对齐时间
            idx_map = resample_indices(t_in, times)

            collision = False
            for k in win_idx:
                x = float(x_in[idx_map[k]]); y = float(y_in[idx_map[k]])
                yaw = float(yaw_in[idx_map[k]])
                ego_poly = oriented_box(x, y, yaw, Lm, Wm)
                if ego_poly.intersects(Ft[k]):
                    collision = True
                    break
            if not collision:
                ok_by_runs = 1
                break

        results.append(dict(sid=sid, mu=mu, v0=v0,
                            ok_by_runs=int(ok_by_runs),
                            ok_by_tube=int(ok_by_tube)))

    # 写出 with_runs
    out_dir = out/"metrics"
    out_csv = out_dir/"s2t_eval_with_runs.csv"
    df = pd.DataFrame(results)
    df.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[07b] wrote with_runs:    {out_csv.resolve()} (rows={len(df)})")

    # 混淆矩阵（只统计 ok_by_tube/ok_by_runs 均在 {0,1} 的行）
    sub = df[df["ok_by_tube"].isin([0,1]) & df["ok_by_runs"].isin([0,1])]
    TP = int(((sub.ok_by_tube==1) & (sub.ok_by_runs==1)).sum())
    TN = int(((sub.ok_by_tube==0) & (sub.ok_by_runs==0)).sum())
    FP = int(((sub.ok_by_tube==1) & (sub.ok_by_runs==0)).sum())
    FN = int(((sub.ok_by_tube==0) & (sub.ok_by_runs==1)).sum())
    prec = TP / (TP+FP+1e-9)
    rec  = TP / (TP+FN+1e-9)
    acc  = (TP+TN) / max(len(sub),1)

    conf_csv = out_dir/"s2t_confusion_matrix.csv"
    pd.DataFrame([dict(TP=TP,TN=TN,FP=FP,FN=FN,
                       precision=prec,recall=rec,accuracy=acc,N=len(sub))])\
      .to_csv(conf_csv, index=False, encoding="utf-8")
    print(f"[07b] wrote confusion:    {conf_csv.resolve()}")
    print(f"[07b] N={len(sub)}, TP={TP}, TN={TN}, FP={FP}, FN={FN}, "
          f"precision={prec:.3f}, recall={rec:.3f}, accuracy={acc:.3f}")

if __name__ == "__main__":
    main()
