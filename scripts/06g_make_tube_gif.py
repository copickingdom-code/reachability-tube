# scripts/06g_make_tube_gif.py
# Build time-lapse GIFs per (mu,v0) from tube_layered_{mu}_{v0}.json
from pathlib import Path
import json, math
import imageio.v2 as imageio
import matplotlib.pyplot as plt
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union

try:
    from vehicletube.paths import outputs_root
except Exception:
    def outputs_root(): return Path("outputs")

def parse_mu_v0(stem: str):
    # stem example: "tube_layered_0.3_30"
    parts = stem.split("_")
    mu = parts[-2]
    v0 = parts[-1]
    return mu, v0

def union_slice_polys(slice_obj):
    polys = []
    for b in slice_obj.get("bins", []):
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
                polys += [gg for gg in getattr(g, "geoms", [])
                          if (gg and (not gg.is_empty) and gg.geom_type == "Polygon")]
        except Exception:
            continue
    if not polys:
        return None
    try:
        U = unary_union(polys)
        if U.is_empty:
            return None
        # pick largest polygon if MultiPolygon
        if U.geom_type != "Polygon":
            parts = list(U.geoms)
            parts.sort(key=lambda g: g.area, reverse=True)
            U = parts[0]
        return U
    except Exception:
        return None

def render_frames(tube_json: Path, out_dir: Path):
    obj = json.loads(tube_json.read_text(encoding="utf-8"))
    layered = obj.get("layered", [])
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for k, s in enumerate(layered):
        A = union_slice_polys(s)
        fig, ax = plt.subplots(figsize=(4.2, 3.4))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, ls=":", lw=0.6)
        ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)")
        ax.set_title(f"t = {k*0.01:.2f} s", fontsize=10)  # 100 Hz
        if A is not None:
            xs, ys = A.exterior.xy
            ax.plot(xs, ys, lw=1.2)
        ax.set_xlim(-15, 60); ax.set_ylim(-15, 15)
        fig.tight_layout()
        png = out_dir / f"frame_{k:03d}.png"
        fig.savefig(png, dpi=140, bbox_inches="tight")
        plt.close(fig)
        frames.append(png)
    return frames

def main():
    root = outputs_root()
    man = root / "manifests"
    gif_dir = root / "gifs"
    gif_dir.mkdir(parents=True, exist_ok=True)
    for j in sorted(man.glob("tube_layered_*.json")):
        mu, v0 = parse_mu_v0(j.stem)
        frame_dir = gif_dir / "frames" / f"mu{mu}_v{v0}"
        frames = render_frames(j, frame_dir)
        if not frames:
            print("[WARN] no frames for", j);
            continue
        gif_path = gif_dir / f"tube_mu{mu}_v{v0}.gif"
        imgs = [imageio.imread(p) for p in frames]
        imageio.mimsave(gif_path, imgs, duration=0.06)  # ~16 fps
        print("Saved", gif_path)

if __name__ == "__main__":
    main()
