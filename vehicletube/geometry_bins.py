import numpy as np
import alphashape
from shapely.geometry import Point, LineString, MultiPoint
import numpy as np, shapely.geometry as geom, shapely.ops as ops
import alphashape

def _poly_from_points(pts, alpha_cfg):
    """
    pts: (N,2) array-like of XY (meters).
    alpha_cfg: dict with keys:
        - value: float | "auto"  （alpha 参数；"auto" 用 optimizealpha）
        - min_pts_alpha: int     （少于此阈值不用 alphashape，改用凸包）
        - buffer_eps_m: float    （退化时的最小缓冲半径，m）
        - min_area_m2: float     （面积太小时加缓冲）
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) == 0:
        return None

    # 去 NaN + 去重
    pts = pts[~np.isnan(pts).any(axis=1)]
    if len(pts) == 0:
        return None
    pts = np.unique(pts, axis=0)

    buf   = float(alpha_cfg.get('buffer_eps_m', 0.05))     # 5 cm
    minA  = float(alpha_cfg.get('min_area_m2', 1e-3))
    thr   = int(alpha_cfg.get('min_pts_alpha', 8))
    aval  = alpha_cfg.get('value', 'auto')

    # 1 点 / 2 点：用点或线的 buffer
    if len(pts) == 1:
        return Point(pts[0]).buffer(buf)
    if len(pts) == 2:
        return LineString(pts).buffer(buf)

    # 共线：秩 < 2 → 几何上是条线；用凸包后再 buffer
    if np.linalg.matrix_rank(pts - pts.mean(axis=0)) < 2:
        return MultiPoint(pts).convex_hull.buffer(buf)

    # 点数足够→优先 alphashape；否则用凸包
    try:
        if len(pts) >= thr:
            if aval is None or str(aval).lower() == "auto":
                try:
                    aval = alphashape.optimizealpha(pts)
                except Exception:
                    aval = 0.0  # 退化用凸包
            shp = alphashape.alphashape(pts, float(aval))
        else:
            shp = MultiPoint(pts).convex_hull
    except Exception:
        shp = MultiPoint(pts).convex_hull

    # 统一保证输出是“面”几何；线/点 → buffer
    if shp.is_empty:
        return None
    if shp.geom_type not in ("Polygon", "MultiPolygon"):
        shp = shp.buffer(buf)
    elif getattr(shp, "area", 0.0) < minA:
        shp = shp.buffer(buf)
    return shp

def build_layered_tube(slices, v_bins, psi_bins, alpha_cfg):
    K = len(slices); result = []
    v_edges = np.array(v_bins, dtype=float)
    psi_edges = np.array(psi_bins, dtype=float)
    for k in range(K):
        sl = slices[k]
        x,y,v,psi = map(np.asarray, (sl['x'],sl['y'],sl['v'],sl['psi']))
        bins_out = []
        if len(x)==0: result.append(dict(time_index=k, bins=bins_out)); continue
        for i in range(len(v_edges)-1):
            for j in range(len(psi_edges)-1):
                mask = (v>=v_edges[i])&(v<v_edges[i+1])&(psi>=psi_edges[j])&(psi<psi_edges[j+1])
                pts = np.c_[x[mask], y[mask]]
                poly = _poly_from_points(pts, alpha_cfg)
                if poly is None or poly.is_empty: continue
                bins_out.append(dict(v_bin=[float(v_edges[i]),float(v_edges[i+1])],
                                     psi_bin=[float(psi_edges[j]),float(psi_edges[j+1])],
                                     area=float(poly.area), wkt=poly.wkt))
        result.append(dict(time_index=k, bins=bins_out))
    return result
