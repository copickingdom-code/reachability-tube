# scripts/08a_compress_and_timing.py
import argparse, json, os, re, glob, time, math
import numpy as np
import pandas as pd
from tqdm import tqdm

# shapely for WKT + geometry ops
from shapely import wkt as shapely_wkt
from shapely.geometry import Polygon, MultiPolygon, Point
from shapely.ops import unary_union
from shapely.prepared import prep
from shapely.validation import make_valid


# ----------------------------- helpers -----------------------------

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def parse_mu_v0_from_name_or_json(path, data):
    """优先用 JSON 内的 mu/v0；没有时再从文件名解析。"""
    mu = data.get('mu', None)
    v0 = data.get('v0', None)
    if mu is not None and v0 is not None:
        return float(mu), int(v0)
    m = re.search(r'([0-9]+\.[0-9])_(\d+)\.json$', os.path.basename(path))
    if m:
        return float(m.group(1)), int(m.group(2))
    return None, None

def polygons_from_layered_json(data):
    """
    解析 layered 结构：
      data['dt']: 采样周期
      data['layered']: 列表，元素含 {'time_index':k, 'bins':[{'wkt': ...}, ...]}
    返回:
      times: (T,)
      polys: list(shapely Polygon) 统一为每时刻的 xy 并集多边形（取最大面积的 polygon）
    """
    dt = float(data.get('dt', 0.01))
    layered = data.get('layered')
    if not layered:
        return None, None

    times, polys = [], []
    for sl in layered:
        ti = sl.get('time_index', 0)
        bins = sl.get('bins', [])
        geoms = []
        for b in bins:
            w = b.get('wkt')
            if not w:
                continue
            try:
                g = shapely_wkt.loads(w)
                if not g.is_empty:
                    geoms.append(g)
            except Exception:
                continue
        if not geoms:
            continue
        u = unary_union(geoms)
        if u.is_empty:
            continue
        u = make_valid(u)
        # 统一为单个 Polygon（MultiPolygon 取最大面积）
        if u.geom_type == 'MultiPolygon':
            u = max(list(u.geoms), key=lambda gg: gg.area)
        elif u.geom_type != 'Polygon':
            u = u.convex_hull
        times.append(ti * dt)
        polys.append(u)
    if not polys:
        return None, None
    return np.array(times, dtype=float), polys

def polygons_from_slices_json(data):
    """
    回退解析：假设每帧提供 xy 点云（dx,dy 或 x,y），我们自行 α-shape 或凸包（此处用凸包兜底）。
    你的工程当前不需要这个分支，但保留以兼容其它数据。
    """
    slices = data.get('slices') or data.get('frames') or data.get('time_slices')
    if not slices:
        return None, None
    # 没有 dt 就用序号当时间
    dt = float(data.get('dt', 0.01))
    times, polys = [], []
    for idx, sl in enumerate(slices):
        t = sl.get('t') or sl.get('time') or sl.get('timestamp') or idx * dt
        pts = sl.get('points') or sl.get('samples') or sl.get('xy') or sl.get('data')
        if not pts:
            continue
        xy = []
        for p in pts:
            if 'dx' in p and 'dy' in p:
                xy.append([p['dx'], p['dy']])
            elif 'x' in p and 'y' in p:
                xy.append([p['x'], p['y']])
        if len(xy) < 3:
            continue
        xy = np.asarray(xy, dtype=float)
        # 兜底用凸包
        try:
            from shapely.geometry import MultiPoint
            poly = MultiPoint([tuple(r) for r in xy]).convex_hull
        except Exception:
            poly = None
        if poly and not poly.is_empty:
            if poly.geom_type == 'MultiPolygon':
                poly = max(list(poly.geoms), key=lambda gg: gg.area)
            times.append(float(t))
            polys.append(poly)
    if not polys:
        return None, None
    return np.array(times, dtype=float), polys

def sample_boundary(poly, n=256):
    """沿外边界按弧长等距采样 n 个点；返回 (n,2) 数组。"""
    coords = np.asarray(poly.exterior.coords)
    segs = np.diff(coords, axis=0)
    lens = np.linalg.norm(segs, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    L = cum[-1]
    if L <= 1e-9:
        return None
    s = np.linspace(0, L, n, endpoint=False)
    pts = []
    for si in s:
        idx = np.searchsorted(cum, si, side='right') - 1
        idx = max(0, min(idx, len(segs) - 1))
        t = (si - cum[idx]) / (lens[idx] + 1e-12)
        p = coords[idx] + t * segs[idx]
        pts.append(p)
    return np.asarray(pts, dtype=float)

def best_cyclic_alignment(A, B, max_shifts=64):
    """循环平移 B 以最小化与 A 的均方误差。"""
    n = A.shape[0]
    step = max(1, n // max_shifts)
    best_s, best_val = 0, 1e18
    for s in range(0, n, step):
        Br = np.roll(B, shift=s, axis=0)
        val = np.mean(np.sum((A - Br) ** 2, axis=1))
        if val < best_val:
            best_val, best_s = val, s
    return np.roll(B, shift=best_s, axis=0)

def hausdorff_symmetric(A, B, batch=2048):
    """对称 Hausdorff（欧氏距离）。"""
    def directed(P, Q):
        dmax = 0.0
        for i in range(0, len(P), batch):
            Pchunk = P[i:i+batch]
            d2 = ((Pchunk[:, None, :] - Q[None, :, :]) ** 2).sum(axis=2)
            dmin = np.sqrt(d2.min(axis=1))
            dmax = max(dmax, float(dmin.max()))
        return dmax
    return max(directed(A, B), directed(B, A))

def greedy_kf_thinning(times, samples, eps_h=0.2):
    """
    基于线性插值的关键帧抽稀，直到所有被省略帧的边界与插值边界的 Hausdorff <= eps_h。
    返回：key_indices（升序），Hmax（最终最大误差）
    """
    N = len(times)
    keys = [0, N - 1]

    def interval_max_error(i, j):
        Ai, Aj = samples[i], samples[j]
        Aj_aligned = best_cyclic_alignment(Ai, Aj, max_shifts=64)
        worst, worst_k = 0.0, None
        ti, tj = times[i], times[j]
        for k in range(i + 1, j):
            tau = (times[k] - ti) / (tj - ti + 1e-12)
            Ahat = (1.0 - tau) * Ai + tau * Aj_aligned
            h = hausdorff_symmetric(samples[k], Ahat)
            if h > worst:
                worst, worst_k = h, k
        return worst, worst_k

    changed = True
    while changed:
        changed = False
        new_keys = [keys[0]]
        for a, b in zip(keys, keys[1:]):
            worst, worst_k = interval_max_error(a, b)
            if worst_k is not None and worst > eps_h:
                new_keys.append(worst_k)
                changed = True
            new_keys.append(b)
        keys = sorted(set(new_keys))

    # 计算最终 Hmax（在最终 keys 下的真正最大误差）
    Hmax = 0.0
    for a, b in zip(keys, keys[1:]):
        Ai, Aj = samples[a], samples[b]
        Aj_aligned = best_cyclic_alignment(Ai, Aj, max_shifts=64)
        ti, tj = times[a], times[b]
        for k in range(a + 1, b):
            tau = (times[k] - ti) / (tj - ti + 1e-12)
            Ahat = (1.0 - tau) * Ai + tau * Aj_aligned
            h = hausdorff_symmetric(samples[k], Ahat)
            Hmax = max(Hmax, h)
    return keys, Hmax

def quantize_xy(vertices_list, qbits=12, save_path=None):
    """对关键帧边界顶点做 min-max 归一化 + qbit 量化，保存 npz；返回字节数。"""
    all_pts = np.concatenate(vertices_list, axis=0)
    mins = all_pts.min(axis=0)
    span = np.maximum(all_pts.max(axis=0) - mins, 1e-9)

    normed = [(v - mins) / span for v in vertices_list]
    L = (1 << qbits) - 1
    dtype = np.uint16 if qbits <= 16 else np.uint32
    quants = [np.clip(np.round(v * L), 0, L).astype(dtype) for v in normed]

    pack = {
        'qbits': qbits,
        'mins': mins.astype(np.float32),
        'span': span.astype(np.float32),
        'frames': quants
    }
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        np.savez_compressed(save_path, **pack)
        return os.path.getsize(save_path)
    else:
        total = sum(arr.nbytes for arr in quants) + 2 * mins.nbytes
        return total

def measure_query_time_single_poly(poly, n_points=10000, seed=123):
    """
    在代表性关键帧 polygon 上评估 membership（contains）平均耗时（ms/点）。
    用该 polygon 的外接矩形均匀采样。
    """
    if poly is None or poly.is_empty:
        return float('nan')
    minx, miny, maxx, maxy = poly.bounds
    rng = np.random.default_rng(seed)
    X = rng.uniform(minx, maxx, size=(n_points, 1))
    Y = rng.uniform(miny, maxy, size=(n_points, 1))
    pts = np.hstack([X, Y])

    P = prep(poly)
    t0 = time.perf_counter()
    inside = 0
    for pt in pts:
        if P.contains(Point(pt[0], pt[1])):
            inside += 1
    t1 = time.perf_counter()
    avg_ms = (t1 - t0) * 1000.0 / n_points
    return avg_ms

def build_times_polys(manifest_path):
    """
    自动识别 JSON 结构，返回 (times, polys)。
    优先 layered；失败再回退 slices 结构。
    """
    data = load_json(manifest_path)
    times, polys = polygons_from_layered_json(data)
    if times is None or polys is None:
        times, polys = polygons_from_slices_json(data)
    if times is None or polys is None:
        raise ValueError("JSON does not contain 'layered' nor 'slices/frames/time_slices'.")
    return data, times, polys

# ----------------------------- main -----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifests", type=str, default="outputs/manifests", help="Directory of JSON manifests")
    ap.add_argument("--mu", type=float, nargs="+", default=[0.3,0.5,0.8])
    ap.add_argument("--v0", type=int,   nargs="+", default=[60])
    ap.add_argument("--eps_h_xy", type=float, default=0.2, help="Hausdorff budget (m) on XY")
    ap.add_argument("--qbits", type=int, default=12, help="Quantization bits (per coordinate)")
    ap.add_argument("--outdir", type=str, default="outputs/compression")
    ap.add_argument("--n_query", type=int, default=10000, help="Random points for query timing")
    ap.add_argument("--boundary_samples", type=int, default=256, help="Arc-length samples per boundary")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rows = []

    all_files = sorted(glob.glob(os.path.join(args.manifests, "*.json")))
    if not all_files:
        print(f"[WARN] No JSON in {args.manifests}")
        return

    # 建立 (mu,v0) -> 文件 的索引（优先以 JSON 内的 mu/v0 为准）
    index = {}
    for f in all_files:
        try:
            data = load_json(f)
            mu_i, v0_i = parse_mu_v0_from_name_or_json(f, data)
            if mu_i is None or v0_i is None:
                continue
            index.setdefault((round(mu_i, 3), int(v0_i)), f)
        except Exception:
            continue

    for mu in args.mu:
        for v in args.v0:
            key = (round(mu, 3), int(v))
            f = index.get(key, None)
            if not f:
                print(f"[SKIP] mu={mu}, v0={v}: no matching JSON in {args.manifests}")
                continue
            try:
                data, times, polys = build_times_polys(f)
            except Exception as e:
                print(f"[SKIP] mu={mu}, v0={v}: {e}")
                continue

            # 采样每帧边界
            samples = []
            valid_pairs = []
            for p in polys:
                sb = sample_boundary(p, n=args.boundary_samples)
                if sb is not None:
                    samples.append(sb)
                    valid_pairs.append(True)
                else:
                    valid_pairs.append(False)
            if not any(valid_pairs):
                print(f"[SKIP] mu={mu}, v0={v}: no valid boundaries.")
                continue
            # 对齐 times 与 samples（去掉无效）
            valid_idx = [i for i, ok in enumerate(valid_pairs) if ok]
            times = times[valid_idx]
            samples = [samples[i] for i in range(len(samples))]

            # KF-ε 抽稀
            key_idx, Hmax = greedy_kf_thinning(times, samples, eps_h=args.eps_h_xy)
            K = len(key_idx)

            # 量化+存储估计
            key_vertices = [samples[i].astype(np.float32) for i in key_idx]
            save_path = os.path.join(args.outdir, f"tube_xy_mu{mu}_v{v}_q{args.qbits}.npz")
            nbytes = quantize_xy(key_vertices, qbits=args.qbits, save_path=save_path)
            storage_mb = nbytes / (1024.0 * 1024.0)

            # 选代表帧（关键帧里面积中位数的那一帧）做 contains 耗时
            key_polys = [polys[i] for i in key_idx]
            areas = [kp.area for kp in key_polys]
            mid = np.argsort(areas)[len(areas)//2]
            query_ms = measure_query_time_single_poly(key_polys[int(mid)], n_points=args.n_query)

            rows.append({
                "mu": mu,
                "v0": v,
                "K": K,
                "Storage_MB": round(storage_mb, 4),
                "H_max_xy_m": round(float(Hmax), 4),
                "Query_ms_per_point": round(float(query_ms), 4),
                "eps_H_xy": args.eps_h_xy,
                "qbits": args.qbits,
                "n_query": args.n_query,
                "boundary_samples": args.boundary_samples,
                "source_file": os.path.basename(f)
            })

    if rows:
        df = pd.DataFrame(rows).sort_values(["v0", "mu"])
        out_csv = os.path.join(args.outdir, "compress_metrics.csv")
        df.to_csv(out_csv, index=False)
        print(f"[OK] Saved: {out_csv}")
        print(df.to_string(index=False))
    else:
        print("[WARN] No rows produced. Check manifests / filters / dependencies.")


if __name__ == "__main__":
    main()
