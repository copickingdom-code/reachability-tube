#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json, os, re, glob
from typing import List, Tuple, Dict, Any
import numpy as np
import pandas as pd
import argparse


# ---------- WKT / 几何工具 ----------

def parse_wkt_polygon(wkt: str) -> np.ndarray:
    """
    将单环 POLYGON WKT 解析为 Nx2 的 numpy 数组（顺序为边界点顺时针/逆时针）。
    形如：POLYGON ((x y, x y, ...))
    """
    if not isinstance(wkt, str):
        raise ValueError("WKT must be a string")
    wkt = wkt.strip()
    if not wkt.upper().startswith("POLYGON"):
        raise ValueError(f"Unsupported WKT (not POLYGON): {wkt[:30]}...")
    start = wkt.find("((");
    end = wkt.rfind("))")
    if start == -1 or end == -1 or end <= start + 2:
        raise ValueError(f"Malformed POLYGON WKT: {wkt[:50]}...")
    inner = wkt[start + 2:end]
    coords = []
    for token in inner.split(","):
        token = token.strip()
        parts = token.split()
        if len(parts) < 2:
            parts = re.split(r"[ ,]+", token.strip())
        if len(parts) < 2:
            continue
        x = float(parts[0]);
        y = float(parts[1])
        coords.append((x, y))
    if len(coords) < 3:
        raise ValueError(f"Not enough points parsed from WKT: {wkt[:60]}...")
    return np.array(coords, dtype=float)


def _directed_hausdorff(A: np.ndarray, B: np.ndarray) -> float:
    # 逐块计算，避免大矩阵一次性占用内存
    block = 2048
    max_min = 0.0
    for i in range(0, A.shape[0], block):
        a = A[i:i + block]
        diff = a[:, None, :] - B[None, :, :]
        d2 = np.sum(diff * diff, axis=2)
        min_d = np.sqrt(np.min(d2, axis=1))
        max_min = max(max_min, float(np.max(min_d)))
    return max_min


def hausdorff_distance(A: np.ndarray, B: np.ndarray) -> float:
    """
    对称 Hausdorff 距离：max( sup_a inf_b d(a,b), sup_b inf_a d(a,b) )
    A, B 为 Nx2 / Mx2 点集。
    """
    if A.ndim != 2 or B.ndim != 2 or A.shape[1] != 2 or B.shape[1] != 2:
        raise ValueError("A and B must be Nx2 and Mx2 arrays")
    return max(_directed_hausdorff(A, B), _directed_hausdorff(B, A))


def polygon_perimeter(pts: np.ndarray) -> float:
    if len(pts) < 2:
        return 0.0
    diff = np.diff(np.vstack([pts, pts[0]]), axis=0)
    seg_len = np.sqrt((diff ** 2).sum(axis=1))
    return float(seg_len.sum())


def polygon_centroid(pts: np.ndarray) -> Tuple[float, float]:
    # Shoelace 质心；退化时回退为点均值
    x = pts[:, 0];
    y = pts[:, 1]
    x2 = np.append(x, x[0]);
    y2 = np.append(y, y[0])
    cross = x2[:-1] * y2[1:] - x2[1:] * y2[:-1]
    area2 = cross.sum()
    if abs(area2) < 1e-12:
        return float(np.mean(x)), float(np.mean(y))
    cx = ((x2[:-1] + x2[1:]) * cross).sum() / (3 * area2)
    cy = ((y2[:-1] + y2[1:]) * cross).sum() / (3 * area2)
    return float(cx), float(cy)


# ---------- 数据处理 ----------

def dataset_id_from_path(path: str) -> str:
    base = os.path.basename(path)
    name, _ = os.path.splitext(base)
    return name


def load_json_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert "layered" in data and isinstance(data["layered"], list), "JSON missing 'layered'"
    return data


def process_file(path: str) -> Dict[str, pd.DataFrame]:
    data = load_json_dataset(path)
    dataset_id = dataset_id_from_path(path)
    mu = data.get("mu", None)
    v0 = data.get("v0", None)
    dt = data.get("dt", None)
    horizon = data.get("horizon", None)

    vol_rows, raw_rows, sliced_rows = [], [], []
    bins_by_time: Dict[int, Dict[Tuple[float, float, float, float], Dict[str, Any]]] = {}

    for layer in data["layered"]:
        t_idx = int(layer["time_index"])
        bins = layer.get("bins", [])
        total_area = sum(float(b.get("area", 0.0)) for b in bins)
        t = (t_idx * dt) if isinstance(dt, (int, float)) else t_idx

        vol_rows.append({
            "dataset": dataset_id, "mu": mu, "v0": v0, "dt": dt, "horizon": horizon,
            "time_index": t_idx, "t": t, "n_bins": len(bins), "total_area": total_area
        })

        bins_dict = {}
        # 先收集本层的 raw & sliced
        layer_raw = []
        for b in bins:
            vbin = b.get("v_bin", [None, None])
            psibin = b.get("psi_bin", [None, None])
            area = float(b.get("area", 0.0))
            wkt = b.get("wkt", None)
            key = (float(vbin[0]), float(vbin[1]), float(psibin[0]), float(psibin[1]))

            pts = None;
            cx = cy = perim = np.nan
            if isinstance(wkt, str):
                try:
                    pts = parse_wkt_polygon(wkt)
                    cx, cy = polygon_centroid(pts)
                    perim = polygon_perimeter(pts)
                except Exception:
                    pass
            row = {
                "dataset": dataset_id, "mu": mu, "v0": v0, "dt": dt,
                "time_index": t_idx, "t": t,
                "v_lo": vbin[0], "v_hi": vbin[1], "psi_lo": psibin[0], "psi_hi": psibin[1],
                "area": area, "centroid_x": cx, "centroid_y": cy, "perimeter": perim
            }
            layer_raw.append(row)
            bins_dict[key] = {"wkt": wkt, "pts": pts, "area": area}

        # sliced_metrics：加 area_frac
        for r in layer_raw:
            frac = (r["area"] / total_area) if total_area > 0 else np.nan
            sliced_rows.append({**r, "area_frac": frac})
        raw_rows.extend(layer_raw)
        bins_by_time[t_idx] = bins_dict

    # Hausdorff：与前一帧同 key 的多边形比
    h_rows = []
    times_sorted = sorted(bins_by_time.keys())
    for i in range(1, len(times_sorted)):
        t_idx = times_sorted[i]
        prev_idx = times_sorted[i - 1]
        t = (t_idx * dt) if isinstance(dt, (int, float)) else t_idx
        current = bins_by_time[t_idx]
        prev = bins_by_time[prev_idx]

        dists = []
        for key, cur in current.items():
            if key in prev:
                A = cur.get("pts", None)
                B = prev[key].get("pts", None)
                if A is not None and B is not None:
                    try:
                        d = hausdorff_distance(A, B)
                        dists.append(float(d))
                    except Exception:
                        pass
        if len(dists) == 0:
            mean_h = max_h = p95_h = np.nan
        else:
            arr = np.array(dists, dtype=float)
            mean_h = float(np.mean(arr))
            max_h = float(np.max(arr))
            p95_h = float(np.percentile(arr, 95))

        h_rows.append({
            "dataset": dataset_id, "mu": mu, "v0": v0, "dt": dt, "horizon": horizon,
            "time_index": t_idx, "t": t,
            "pairs": len(dists), "mean_hausdorff": mean_h, "p95_hausdorff": p95_h, "max_hausdorff": max_h
        })

    return {
        "volume_vs_dt": pd.DataFrame(vol_rows),
        "volume_vs_dt_raw": pd.DataFrame(raw_rows),
        "sliced_metrics": pd.DataFrame(sliced_rows),
        "hausdorff_vs_dt": pd.DataFrame(h_rows),
    }


def process_many(input_glob: str, out_dir: str) -> Dict[str, pd.DataFrame]:
    files = sorted(glob.glob(input_glob))
    if not files:
        raise FileNotFoundError(f"No JSON files matched: {input_glob}")
    os.makedirs(out_dir, exist_ok=True)

    all_dfs = {name: [] for name in ["volume_vs_dt", "volume_vs_dt_raw", "sliced_metrics", "hausdorff_vs_dt"]}
    for f in files:
        if not f.lower().endswith(".json"):
            continue
        try:
            dfs = process_file(f)
            for k in all_dfs:
                all_dfs[k].append(dfs[k])
        except Exception as e:
            print(f"[WARN] Failed to process {f}: {e}")

    out = {k: (pd.concat(v, ignore_index=True) if v else pd.DataFrame()) for k, v in all_dfs.items()}
    # 写 CSV
    for name, df in out.items():
        csv_path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(csv_path, index=False)
        print(f"[OK] Saved: {csv_path}  ({len(df):,} rows)")
    return out


# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(description="Compute convergence metrics from layered JSONs.")
    parser.add_argument("--input", default="outputs/manifests/*.json", help="Glob for input JSON files (default: *.json)")
    parser.add_argument("--out", default="outputs/convergence_out", help="Output directory for CSVs (default: metrics_out)")
    args = parser.parse_args()
    process_many(args.input, args.out)


if __name__ == "__main__":
    main()
