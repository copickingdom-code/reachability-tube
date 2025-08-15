# -*- coding: utf-8 -*-
"""
07a — Scenario-to-Tube Geometry Check (Layer 1)
- 输入：
  - outputs/manifests/tube_layered_{mu}_{v0}.json   （4D 分箱后每个切片的 XY 轮廓集合，已有）
  - data/scenarios/scenarios_minimal.csv            （我们生成/手写的长尾微场景）
  - configs/default.yaml（可选）：
      dt_s, horizon_s,
      s2t: {area_eps_m2, frac_thr, roi_radius_m}
      tube: {erosion_m, min_area_m2}
- 输出：
  - outputs/metrics/s2t_eval_geometry.csv
  - outputs/figures/s2t/{sid}_mu{mu}_v{v0}.png     （每个场景 4 张帧叠图）
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from shapely.geometry import Polygon, Point
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union
from shapely.affinity import translate

# ==== 路径/配置（兼容无 vehicletube 包的情况）====
try:
    from vehicletube.paths import outputs_root
    from vehicletube.config import load as cfg_load
except Exception:
    def outputs_root() -> Path:
        return Path("outputs")
    def cfg_load(_path: str):
        return {}

# ---------------- helpers ----------------
def _log_header(title: str):
    print("=" * 8, title, "=" * 8)

def load_cfg():
    cfg = cfg_load("configs/default.yaml") or {}
    dt = float(cfg.get("dt_s", 0.01))
    T  = float(cfg.get("horizon_s", 2.0))
    s2t = cfg.get("s2t", {}) or {}
    area_eps   = float(s2t.get("area_eps_m2", 0.20))
    frac_thr   = float(s2t.get("frac_thr", 0.90))
    roi_radius = float(s2t.get("roi_radius_m", 15.0))
    tube = cfg.get("tube", {}) or {}
    erosion_m    = float(tube.get("erosion_m", 0.0))     # 负 buffer 内缩（m）
    min_area_m2  = float(tube.get("min_area_m2", 0.02))  # 过滤面积（m^2）
    return dt, T, area_eps, frac_thr, roi_radius, erosion_m, min_area_m2

def _robust_union_and_erode(polys, erosion_m: float, min_area_m2: float):
    """
    对一个切片的多边形做并集 + 内缩（可选）+ 过滤：
      - union 后 buffer(0) 清理
      - 负 buffer 自适应步长（避免蚀空）
      - MultiPolygon 仅保留最大面积那片
    返回：Polygon 或 None
    """
    if not polys:
        return None
    try:
        U = unary_union(polys)
    except Exception:
        return None
    if U.is_empty:
        return None
    # 清 invalid
    try:
        if not U.is_valid:
            U = U.buffer(0)
    except Exception:
        pass
    if U.is_empty:
        return None

    # 自适应蚀刻（负 buffer）
    if erosion_m > 0.0:
        # 限制单步不超过 bbox 最小边长的 1/4，防止一步蚀空
        xmin, ymin, xmax, ymax = U.bounds
        min_dim = max(1e-6, min(xmax - xmin, ymax - ymin))
        step = min(erosion_m, 0.25 * min_dim)

        V = U
        s = step
        while s > 1e-4:
            try:
                W = V.buffer(-s, join_style=2)  # mitre corners，保持边界形状
                if (not W.is_empty) and (W.area >= min_area_m2):
                    if not W.is_valid:
                        W = W.buffer(0)
                    if (not W.is_empty) and (W.area >= min_area_m2):
                        V = W
                        break  # 找到合适的蚀刻结果
            except Exception:
                pass
            s *= 0.5  # 过量 → 减半再试
        U = V  # 如果失败会落回原 U

    if U.is_empty:
        return None

    # MultiPolygon → 过滤小碎片并选最大面
    if U.geom_type == "MultiPolygon":
        parts = [g for g in U.geoms if (not g.is_empty) and (g.area >= min_area_m2)]
        if not parts:
            return None
        parts.sort(key=lambda g: g.area, reverse=True)
        return parts[0]
    elif U.geom_type == "Polygon":
        return U if U.area >= min_area_m2 else None
    else:
        return None

def load_tube_layer(mu: str, v0: str, erosion_m: float, min_area_m2: float):
    """返回切片多边形列表（统一为 Polygon；无则 None）"""
    p = outputs_root() / "manifests" / f"tube_layered_{mu}_{v0}.json"
    if not p.exists():
        return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    slices = []
    for s in obj.get("layered", []):
        polys = []
        for b in s.get("bins", []):
            w = b.get("wkt")
            if not w:
                continue
            try:
                g = shapely_wkt.loads(w)
                if g.is_empty:
                    continue
                if g.geom_type == "Polygon":
                    polys.append(g)
                else:
                    for gg in getattr(g, "geoms", []):
                        if (not gg.is_empty) and (gg.geom_type == "Polygon"):
                            polys.append(gg)
            except Exception:
                continue

        U = _robust_union_and_erode(polys, erosion_m, min_area_m2)
        slices.append(U if U is not None else None)
    return slices

def make_obstacle_polygon(kind: str, x: float, y: float, L: float, W: float, R: float, margin: float):
    if kind == "ped_xing":
        base = Point(x, y).buffer(max(R, 0.4))
    else:
        # 车辆矩形（轴对齐），后面只做平移，不旋转（障碍物默认不旋转）
        hw, hl = W / 2.0, L / 2.0
        base = Polygon([(x - hl, y - hw), (x + hl, y - hw), (x + hl, y + hw), (x - hl, y + hw)])
    return base.buffer(max(margin, 0.0))

def scenario_to_Ft(row: pd.Series, times: np.ndarray):
    """生成每个时刻障碍物占据 F_t（Shapely 多边形）"""
    kind = str(row.get("kind", "cutin"))
    x0 = float(row.get("x0", 12.0)); y0 = float(row.get("y0", -3.5))
    vx = float(row.get("vx",  0.5)); vy = float(row.get("vy",  1.5))
    L  = float(row.get("L_obs", 4.5)); W = float(row.get("W_obs", 1.8))
    R  = float(row.get("R_obs", 0.0))
    margin = float(row.get("margin_m", 0.6))
    base = make_obstacle_polygon(kind, 0.0, 0.0, L, W, R, margin)
    Ft = [translate(base, xoff=x0 + vx * t, yoff=y0 + vy * t) for t in times]
    return Ft

def eval_one(row: pd.Series, tube_slices, times: np.ndarray,
             area_eps: float, frac_thr: float, roi_radius: float,
             save_overlay: bool = True):
    """
    判定规则（严格）：
      - 仅在“风险窗口”评估：障碍与原点距离 <= roi_radius 的切片集合 W
      - 令 S_k = A_k \ F_k 的面积；frac_ok = |{k∈W : S_k > area_eps}| / |W|
      - ok = 1 当且仅当 frac_ok >= frac_thr 且 min(S_k) > 0
    """
    if tube_slices is None:
        return 0, 0.0, 0.0, None, 0.0

    Ft = scenario_to_Ft(row, times)
    M = min(len(tube_slices), len(Ft))
    times = times[:M]; Ft = Ft[:M]

    origin = Point(0.0, 0.0)
    win_idx = [k for k in range(M) if Ft[k].distance(origin) <= roi_radius]
    if not win_idx:
        return 0, 0.0, 0.0, None, 0.0

    areas = []
    for k in win_idx:
        A = tube_slices[k]; F = Ft[k]
        if (A is None) or (F is None):
            areas.append(0.0); continue
        if not A.is_valid:
            A = A.buffer(0)
        S = A.difference(F)
        areas.append(S.area if (S and not S.is_empty) else 0.0)

    areas = np.asarray(areas, dtype=float)
    frac_ok  = float((areas > area_eps).mean())
    min_area = float(areas.min()) if areas.size else 0.0
    ok = int((frac_ok >= frac_thr) and (min_area > 1e-9))
    best_area = float(areas.max()) if areas.size else 0.0

    fig = None
    if save_overlay:
        picks = np.linspace(0, len(win_idx) - 1, 4, dtype=int)
        fig, axs = plt.subplots(1, len(picks), figsize=(10.5, 2.8), sharey=True)
        for ax, idx in zip(axs, picks):
            k = win_idx[idx]
            A = tube_slices[k]; F = Ft[k]
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"t={times[k]:.2f}s\narea={areas[idx]:.2f} m²", fontsize=9)
            ax.set_xlabel("X (m)")
            if idx == picks[0]:
                ax.set_ylabel("Y (m)")
            ax.grid(True, ls=':', lw=0.6)
            if A is not None:
                xs, ys = A.exterior.xy
                ax.plot(xs, ys, lw=1.2)
            if F is not None:
                xs, ys = F.exterior.xy
                ax.fill(xs, ys, alpha=0.35, lw=0.8)
        fig.tight_layout()

    return ok, best_area, frac_ok, fig, min_area

# ---------------- main ----------------
def main():
    dt, T, area_eps, frac_thr, roi_radius, erosion_m, min_area_m2 = load_cfg()
    scen_csv = Path("data/scenarios/scenarios_minimal.csv")
    out = outputs_root()
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    (out / "figures" / "s2t").mkdir(parents=True, exist_ok=True)

    _log_header("07a — Geometry Eval")
    print(f"[07a] scenarios:   {scen_csv.resolve()}")
    print(f"[07a] write metrics:{(out/'metrics'/'s2t_eval_geometry.csv').resolve()}")
    print(f"[07a] write figs:   {(out/'figures'/'s2t').resolve()}")
    print(f"[07a] params: dt={dt}s, T={T}s, area_eps={area_eps}m², frac_thr={frac_thr}, roi={roi_radius}m, "
          f"erosion={erosion_m}m, min_area={min_area_m2}m²")

    rows = pd.read_csv(scen_csv, dtype={"sid": str}, encoding="utf-8-sig")
    cache = {}  # (mu,v0) -> (slices, times)
    results = []

    for _, r in rows.iterrows():
        mu = str(r.get("mu"))
        v0 = str(r.get("v0_kph"))
        sid = str(r.get("sid"))

        if (mu, v0) not in cache:
            slices = load_tube_layer(mu, v0, erosion_m, min_area_m2)
            if slices is None:
                print(f"[WARN] tube_layered_{mu}_{v0}.json not found — sid={sid} will be ok=0")
                cache[(mu, v0)] = (None, None)
            else:
                times = np.arange(len(slices), dtype=float) * dt
                cache[(mu, v0)] = (slices, times)

        tube_slices, times = cache[(mu, v0)]
        ok, best_area, frac_ok, fig, min_area = eval_one(
            r, tube_slices, times,
            area_eps=area_eps, frac_thr=frac_thr, roi_radius=roi_radius,
            save_overlay=True
        )
        results.append(dict(
            sid=sid, mu=mu, v0=v0,
            ok=int(ok), best_area=best_area,
            frac_ok=frac_ok, min_area=min_area
        ))

        if fig is not None:
            fig.savefig(out / "figures" / "s2t" / f"{sid}_mu{mu}_v{v0}.png",
                        dpi=230, bbox_inches="tight")
            plt.close(fig)

    df = pd.DataFrame(results)
    df.to_csv(out / "metrics" / "s2t_eval_geometry.csv", index=False, encoding="utf-8")
    print("Saved:", (out / "metrics" / "s2t_eval_geometry.csv").resolve())

if __name__ == "__main__":
    main()
