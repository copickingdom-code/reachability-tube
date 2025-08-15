import json, shapely.wkt as wkt, shapely.ops as ops, matplotlib.pyplot as plt
from vehicletube.paths import outputs_root
from vehicletube.config import load
import sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


def plot_slices(mu, v0, times_s=(0.5, 1.0, 1.5, 2.0)):
    out = outputs_root();
    cfg = load("configs/default.yaml");
    dt = float(cfg["dt_s"])
    p = out / "manifests" / f"tube_layered_{mu}_{v0}.json"
    if not p.exists(): print("[WARN]", p, "not found"); return
    obj = json.loads(open(p, "r", encoding="utf-8").read());
    layered = obj["layered"]
    idxs = [max(0, min(int(round(t / dt)), len(layered) - 1)) for t in times_s]

    plt.figure()
    for t, k in zip(times_s, idxs):
        bins = layered[k]["bins"]
        polys = [wkt.loads(b["wkt"]) for b in bins if "wkt" in b]
        if not polys: continue
        union = ops.unary_union(polys)
        geoms = list(union.geoms) if union.geom_type == "MultiPolygon" else [union]
        for g in geoms:
            x, y = g.exterior.xy;
            plt.plot(x, y, label=f"t={t:.1f}s")
    plt.axis('equal');
    plt.grid(True);
    plt.xlabel("X (m)");
    plt.ylabel("Y (m)")
    plt.title(f"Slices overlay μ={mu}, v0={v0} km/h")
    out_png = out / "figures" / f"slices_overlay_mu{mu}_v{v0}.png"
    plt.tight_layout();
    plt.savefig(out_png);
    plt.close();
    print("saved", out_png)


def main():
    cfg = load("configs/default.yaml")
    for mu in cfg["mus"]:
        for v0 in cfg["v0_kph"]:
            try:
                plot_slices(mu, v0)
            except Exception as e:
                print("[ERR]", mu, v0, e)


if __name__ == "__main__":
    main()
