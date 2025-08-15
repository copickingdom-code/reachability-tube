# scripts/06g_anim_tube.py  (polished)
from pathlib import Path
import sys, json, io
import numpy as np
import shapely.wkt as wkt
import shapely.ops as ops
from shapely.geometry import LineString
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import matplotlib as mpl

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from vehicletube.paths import outputs_root
from vehicletube.config import load

plt.rcParams.update({
    "figure.dpi": 160, "font.size": 10,
    "axes.labelsize": 10, "axes.titlesize": 12,
    "xtick.labelsize": 9, "ytick.labelsize": 9,
})

def _union_polygon(bins):
    polys = [wkt.loads(b.get("wkt","")) for b in bins if "wkt" in b]
    if not polys: return None
    U = ops.unary_union(polys)
    if U.is_empty: return None
    if U.geom_type == "Polygon": return U
    parts = list(U.geoms); parts.sort(key=lambda g:g.area, reverse=True)
    return parts[0]

def _ring(poly, n=100):
    ring = LineString(list(poly.exterior.coords))
    if ring.length <= 1e-6: return None
    ts = np.linspace(0,1,n,endpoint=False)
    pts = [ring.interpolate(t, normalized=True) for t in ts]
    xs = np.array([p.x for p in pts]); ys = np.array([p.y for p in pts])
    j = int(np.argmax(xs)); xs = np.roll(xs, -j); ys = np.roll(ys, -j)
    return xs, ys

def _collect(mu, v0, stride=2, n_samples=100):
    out = outputs_root(); cfg = load("configs/default.yaml"); dt=float(cfg["dt_s"])
    p = out/"manifests"/f"tube_layered_{mu}_{v0}.json"
    obj = json.loads(p.read_text(encoding="utf-8"))
    layered = obj["layered"]
    times, rings = [], []
    for k, slc in enumerate(layered):
        if k % stride: continue
        poly = _union_polygon(slc["bins"])
        rng  = _ring(poly, n_samples) if poly else None
        if rng is None: continue
        times.append(k*dt); rings.append(rng)
    return np.array(times,float), rings

def _render(ax, times, rings, upto_idx, elev=24, azim=-55,
            surface_alpha=0.18, wire_every=6, cmap="cividis"):
    ax.clear()
    tmin, tmax = float(times.min()), float(times.max())
    norm = mpl.colors.Normalize(vmin=tmin, vmax=tmax)
    cm   = mpl.cm.get_cmap(cmap)

    # 预留右侧，与静态图一致
    ax.figure.subplots_adjust(left=0.08, right=0.84, bottom=0.10, top=0.90)
    try: ax.set_proj_type('ortho')
    except Exception: pass

    # 环线
    for idx in range(min(upto_idx+1, len(rings))):
        t = times[idx]; xs, ys = rings[idx]
        if idx % wire_every == 0:
            ax.plot(xs, ys, np.full_like(xs, t), lw=0.8, color=cm(norm(t)))

    # 皮肤
    faces, cols = [], []
    for k in range(min(upto_idx, len(rings)-2)+1):
        xs0, ys0 = rings[k]; xs1, ys1 = rings[k+1]
        n = min(len(xs0), len(xs1)); col = cm(norm(0.5*(times[k]+times[k+1])))
        for i in range(n):
            j = (i+1) % n
            faces.append([
                (xs0[i], ys0[i], times[k]),
                (xs0[j], ys0[j], times[k]),
                (xs1[j], ys1[j], times[k+1]),
                (xs1[i], ys1[i], times[k+1]),
            ]); cols.append(col)
    if faces:
        mesh = Poly3DCollection(faces, facecolors=cols, linewidths=0.15,
                                edgecolors=(0,0,0,0.08), alpha=surface_alpha)
        ax.add_collection3d(mesh)

    xs_all = np.concatenate([r[0] for r in rings[:upto_idx+1]])
    ys_all = np.concatenate([r[1] for r in rings[:upto_idx+1]])
    ax.set_xlim(xs_all.min(), xs_all.max())
    ax.set_ylim(ys_all.min(), ys_all.max())
    ax.set_zlim(tmin, tmax)
    ax.view_init(elev=elev, azim=azim)
    ax.set_xlabel('X (m)', labelpad=8); ax.set_ylabel('Y (m)', labelpad=8); ax.set_zlabel('t (s)', labelpad=10)
    ax.set_title('3D capability tube', pad=12)

def make_gif_grow(mu, v0, stride=2, n_samples=100, step=1, duration_ms=70):
    times, rings = _collect(mu, v0, stride, n_samples)
    if len(rings) < 2: return
    fig = plt.figure(figsize=(7.6, 5.2)); ax = fig.add_subplot(111, projection='3d')
    frames=[]
    for k in range(1, len(rings), step):
        _render(ax, times, rings, k)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160); buf.seek(0)
        frames.append(Image.open(buf).convert("P"))
    plt.close(fig)
    outp = outputs_root()/"figures"/f"tube3d_grow_mu{mu}_v{v0}.gif"
    frames[0].save(outp, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print("saved", outp)

def make_gif_orbit(mu, v0, stride=2, n_samples=100,
                   n_frames=64, elev=24, azim0=-55, azim_span=300, duration_ms=60):
    times, rings = _collect(mu, v0, stride, n_samples)
    if len(rings) < 2: return
    fig = plt.figure(figsize=(7.6, 5.2)); ax = fig.add_subplot(111, projection='3d')
    _render(ax, times, rings, len(rings)-1)
    frames=[]
    for i in range(n_frames):
        az = azim0 + azim_span * i/(n_frames-1)
        ax.view_init(elev=elev, azim=az)
        buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=160); buf.seek(0)
        frames.append(Image.open(buf).convert("P"))
    plt.close(fig)
    outp = outputs_root()/"figures"/f"tube3d_orbit_mu{mu}_v{v0}.gif"
    frames[0].save(outp, save_all=True, append_images=frames[1:], duration=duration_ms, loop=0)
    print("saved", outp)

def main():
    cfg = load("configs/default.yaml")
    for mu in cfg["mus"]:
        for v0 in cfg["v0_kph"]:
            make_gif_grow(mu, v0, stride=2, n_samples=100, step=1, duration_ms=70)
            make_gif_orbit(mu, v0, stride=2, n_samples=100, n_frames=64, duration_ms=60)

if __name__ == "__main__":
    main()
