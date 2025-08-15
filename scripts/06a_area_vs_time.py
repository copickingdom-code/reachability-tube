from pathlib import Path
import json, numpy as np, pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union

try:
    from vehicletube.paths import outputs_root
    from vehicletube.config import load as cfg_load
except Exception:
    def outputs_root(): return Path("outputs")
    def cfg_load(_): return {}

def union_area_of_slice(slice_obj, erosion_m=0.0):
    polys=[]
    for b in slice_obj.get("bins", []):
        w=b.get("wkt");
        if not w: continue
        g=shapely_wkt.loads(w)
        if g.is_empty: continue
        if g.geom_type=="Polygon": polys.append(g)
        else:
            polys += [gg for gg in getattr(g,"geoms",[]) if gg.geom_type=="Polygon" and not gg.is_empty]
    if not polys: return 0.0
    U = unary_union(polys)
    if erosion_m>0.0 and (not U.is_empty):
        U = U.buffer(-erosion_m)
    return U.area if (U and not U.is_empty) else 0.0

def handle(mu, v0, erosion_m):
    root = outputs_root()
    p = root/"manifests"/f"tube_layered_{mu}_{v0}.json"
    if not p.exists(): return
    obj = json.loads(p.read_text(encoding="utf-8"))
    areas=[]
    for k,sl in enumerate(obj.get("layered", [])):
        areas.append(dict(mu=mu, v0=v0, k=k, area=union_area_of_slice(sl, erosion_m)))
    df = pd.DataFrame(areas)
    out = root/"metrics"/f"area_union_mu{mu}_v{v0}.csv"
    df.to_csv(out, index=False, encoding="utf-8")
    print("Saved", out)

def main():
    cfg = cfg_load("configs/default.yaml") or {}
    erosion_m = float((cfg.get("tube") or {}).get("erosion_m", 0.0))
    root = outputs_root()/ "manifests"
    if not root.exists(): return
    for j in root.glob("tube_layered_*.json"):
        s=j.stem.split("_")
        mu, v0 = s[-2], s[-1]
        handle(mu, v0, erosion_m)

if __name__=="__main__":
    main()
