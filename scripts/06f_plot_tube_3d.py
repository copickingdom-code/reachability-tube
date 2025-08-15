# scripts/06f_plot_tube_3d.py
from pathlib import Path
import sys, json
import numpy as np
import shapely.wkt as wkt
import shapely.ops as ops
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # 新增：用于放置不遮挡的colorbar

# 让 vehicletube 包可导入
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicletube.paths import outputs_root
from vehicletube.config import load

# 统一风格
plt.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
})

def _union_polygon_of_slice(slice_bins):
    polys = [wkt.loads(b["wkt"]) for b in slice_bins if "wkt" in b]
    if not polys: return None
    U = ops.unary_union(polys)
    if U.is_empty: return None
    if U.geom_type == "Polygon":
        return U
    parts = list(U.geoms)
    parts.sort(key=lambda g: g.area, reverse=True)
    return parts[0]

def _sample_ring(poly, n=120):
    ring = LineString(list(poly.exterior.coords))
    L = ring.length
    if L <= 1e-6: return None
    ts = np.linspace(0.0, 1.0, n, endpoint=False)
    pts = [ring.interpolate(t, normalized=True) for t in ts]
    xs = np.array([p.x for p in pts]); ys = np.array([p.y for p in pts])
    j = int(np.argmax(xs))  # 以 max-x 对齐，防止相邻切片扭绞
    xs = np.roll(xs, -j); ys = np.roll(ys, -j)
    return xs, ys

def _axes_equal_3d(ax, pad=0.05):
    # 等比例 3D 轴（matplotlib 默认不是等比例）
    xlim = ax.get_xlim3d(); ylim = ax.get_ylim3d(); zlim = ax.get_zlim3d()
    xr = xlim[1]-xlim[0]; yr = ylim[1]-ylim[0]; zr = zlim[1]-zlim[0]
    r = max(xr, yr, zr)
    cx = np.mean(xlim); cy = np.mean(ylim); cz = np.mean(zlim)
    ax.set_xlim3d(cx - (1+pad)*r/2, cx + (1+pad)*r/2)
    ax.set_ylim3d(cy - (1+pad)*r/2, cy + (1+pad)*r/2)
    ax.set_zlim3d(cz - (1+pad)*r/2, cz + (1+pad)*r/2)

def build_one(mu, v0, n_samples=120, stride=2, surface_alpha=0.18,
              wire_every=6, cmap_name="viridis"):
    out = outputs_root()
    cfg = load("configs/default.yaml")
    dt  = float(cfg["dt_s"])
    path = out/"manifests"/f"tube_layered_{mu}_{v0}.json"
    if not path.exists():
        print("[WARN]", path, "not found"); return

    obj = json.loads(open(path, "r", encoding="utf-8").read())
    layered = obj["layered"]

    # 收集切片
    times, rings = [], []
    for k, slc in enumerate(layered):
        if k % stride:   # 时序降采样，减少面片数和颜色/文字重叠
            continue
        poly = _union_polygon_of_slice(slc["bins"])
        if poly is None: continue
        samp = _sample_ring(poly, n_samples)
        if samp is None: continue
        times.append(k*dt); rings.append(samp)

    if len(rings) < 2:
        print(f"[WARN] μ={mu}, v0={v0}: not enough slices."); return

    # 颜色按时间映射
    import matplotlib as mpl
    times = np.array(times, dtype=float)
    tmin, tmax = float(times.min()), float(times.max())
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    cmap = mpl.cm.get_cmap(cmap_name)

    fig = plt.figure(figsize=(6.8, 4.6))
    ax  = fig.add_subplot(111, projection='3d')
    try:
        ax.set_proj_type('ortho')  # 正交投影（需要较新版本 matplotlib）
    except Exception:
        pass

    # 画每个时间片的“环线”（稀疏一些）
    for idx, (t, (xs, ys)) in enumerate(zip(times, rings)):
        if (idx % wire_every) == 0:
            ax.plot(xs, ys, np.full_like(xs, t), lw=0.8, color=cmap(norm(t)))

    # 连接相邻时间片形成“皮肤”，面颜色也随时间渐变
    faces, facecols = [], []
    for k in range(len(rings)-1):
        xs0, ys0 = rings[k];   xs1, ys1 = rings[k+1]
        n = min(len(xs0), len(xs1))
        col = cmap(norm(0.5*(times[k]+times[k+1])))
        for i in range(n):
            j = (i+1) % n
            faces.append([
                (xs0[i], ys0[i], times[k]),
                (xs0[j], ys0[j], times[k]),
                (xs1[j], ys1[j], times[k+1]),
                (xs1[i], ys1[i], times[k+1]),
            ])
            facecols.append(col)
    surf = Poly3DCollection(faces, facecolors=facecols, linewidths=0.15,
                            edgecolors=(0,0,0,0.08), alpha=surface_alpha)
    ax.add_collection3d(surf)

    # 轴 & 视角
    ax.set_xlabel('X (m)', labelpad=6)
    ax.set_ylabel('Y (m)', labelpad=6)
    ax.set_zlabel('t (s)', labelpad=6)
    ax.set_title(f'3D capability tube   μ={mu}, v0={v0} km/h', pad=10)
    ax.view_init(elev=24, azim=-55)
    # 自适应范围
    xs_all = np.concatenate([r[0] for r in rings])
    ys_all = np.concatenate([r[1] for r in rings])
    ax.set_xlim(xs_all.min(), xs_all.max())
    ax.set_ylim(ys_all.min(), ys_all.max())
    ax.set_zlim(tmin, tmax)
    _axes_equal_3d(ax, pad=0.05)

    # colorbar（时间 → 颜色），放在轴右侧内嵌区域，避免与 z 轴刻度重叠
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    axin = inset_axes(
        ax, width="3%", height="70%", loc="center left",
        bbox_to_anchor=(1.15, 0., 1, 1),  # 稍微离开主轴
        bbox_transform=ax.transAxes, borderpad=0.0
    )
    cbar = fig.colorbar(mappable, cax=axin)
    cbar.set_label('time (s)')

    fig.tight_layout()

    out_png = out/"figures"/f"tube3d_mu{mu}_v{v0}.png"
    fig.savefig(out_png)   # 仅保存 PNG
    plt.close(fig)
    print("saved", out_png)

def main():
    cfg = load("configs/default.yaml")
    for mu in cfg["mus"]:
        for v0 in cfg["v0_kph"]:
            try:
                build_one(mu, v0, n_samples=120, stride=2,
                          surface_alpha=0.18, wire_every=6, cmap_name="viridis")
            except Exception as e:
                print("[ERR]", mu, v0, e)

if __name__ == "__main__":
    main()
