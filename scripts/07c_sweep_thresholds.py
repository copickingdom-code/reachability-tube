# scripts/07c_sweep_thresholds.py
from pathlib import Path
import json, itertools
import numpy as np, pandas as pd
from shapely import wkt as shapely_wkt
from shapely.ops import unary_union
from shapely.geometry import Point
from shapely.affinity import translate

def outputs_root(): return Path("outputs")

def load_tube_layer(mu, v0, erosion=0.0):
    p = outputs_root()/ "manifests"/ f"tube_layered_{mu}_{v0}.json"
    if not p.exists(): return None
    obj = json.loads(p.read_text(encoding="utf-8"))
    slices=[]
    for s in obj.get("layered", []):
        polys=[]
        for b in s.get("bins", []):
            w=b.get("wkt");
            if not w: continue
            g=shapely_wkt.loads(w)
            if g.is_empty: continue
            if g.geom_type=="Polygon":
                polys.append(g)
            else:
                polys += [gg for gg in getattr(g,"geoms",[]) if gg.geom_type=="Polygon" and not gg.is_empty]
        if not polys:
            slices.append(None); continue
        U = unary_union(polys)
        if U.is_empty: slices.append(None); continue
        if erosion>0.0:
            U = U.buffer(-erosion)
            if U.is_empty: slices.append(None); continue
        slices.append(U if U.geom_type=="Polygon" else max(list(U.geoms), key=lambda g:g.area))
    return slices

def scenario_to_Ft(row, times):
    from shapely.geometry import Polygon, Point
    from shapely.affinity import translate
    kind=str(row["kind"]); x0=row["x0"]; y0=row["y0"]; vx=row["vx"]; vy=row["vy"]
    L=row["L_obs"]; W=row["W_obs"]; R=row["R_obs"]; margin=row["margin_m"]
    if kind=="ped_xing":
        base=Point(0,0).buffer(max(R,0.4))
    else:
        hw,hl=W/2,L/2
        base=Polygon([(-hl,-hw),(hl,-hw),(hl,hw),(-hl,hw)])
    base=base.buffer(max(margin,0.0))
    return [translate(base, x0+vx*t, y0+vy*t) for t in times]

def eval_one(row, tube_slices, dt, area_eps, frac_thr, roi):
    if tube_slices is None: return 0
    K=len(tube_slices); times=np.arange(K)*dt
    Ft=scenario_to_Ft(row,times)
    origin=Point(0,0)
    picks=[k for k in range(K) if Ft[k].distance(origin)<=roi]
    if not picks: return 0
    ok=0; cnt=0
    areas=[]
    for k in picks:
        A=tube_slices[k]; F=Ft[k]
        if (A is None) or (F is None): areas.append(0.0); continue
        S=A.difference(F)
        areas.append(S.area if (S and not S.is_empty) else 0.0)
    areas=np.asarray(areas)
    frac=(areas>area_eps).mean()
    ok = int((frac>=frac_thr) and (areas.min()>1e-9))
    return ok

def main():
    dt=0.01
    scen = pd.read_csv("data/scenarios/scenarios_minimal.csv", dtype={"sid":str})
    with_runs = pd.read_csv(outputs_root()/ "metrics"/ "s2t_eval_with_runs.csv", dtype={"sid":str})
    # ok_by_runs 作为“真值”
    truth = with_runs[["sid","ok_by_runs"]].drop_duplicates()

    # 参数网格（可调）
    AREA=[0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.24, 0.30]
    FRAC=[0.70, 0.80, 0.85, 0.90, 0.95]
    ROI =[3, 6, 8, 10, 12, 14, 16, 18, 20]
    ERO =[0.00, 0.03, 0.05, 0.08, 0.10]

    # tube cache
    cache={}

    rows=[]
    for area_eps, frac_thr, roi, eros in itertools.product(AREA,FRAC,ROI,ERO):
        ok_list=[]
        for _,r in scen.iterrows():
            key=(r["mu"], r["v0_kph"], eros)
            if key not in cache:
                cache[key]=load_tube_layer(r["mu"], r["v0_kph"], erosion=eros)
            ok = eval_one(r, cache[key], dt, area_eps, frac_thr, roi)
            ok_list.append((r["sid"], ok))
        df_ok = pd.DataFrame(ok_list, columns=["sid","ok_by_tube"])
        df = pd.merge(df_ok, truth, on="sid", how="inner")
        TP = int(((df.ok_by_tube==1)&(df.ok_by_runs==1)).sum())
        TN = int(((df.ok_by_tube==0)&(df.ok_by_runs==0)).sum())
        FP = int(((df.ok_by_tube==1)&(df.ok_by_runs==0)).sum())
        FN = int(((df.ok_by_tube==0)&(df.ok_by_runs==1)).sum())
        prec = TP/(TP+FP+1e-9); rec = TP/(TP+FN+1e-9); acc = (TP+TN)/max(len(df),1)
        rows.append(dict(area_eps=area_eps, frac_thr=frac_thr, roi=roi, erosion=eros,
                         TP=TP,TN=TN,FP=FP,FN=FN,precision=prec,recall=rec,accuracy=acc,N=len(df)))
        print(f"[grid] area={area_eps:.2f} frac={frac_thr:.2f} roi={roi:.0f} eros={eros:.2f} "
              f"→ P={prec:.3f} R={rec:.3f} A={acc:.3f}  (TP={TP},FP={FP})")
    out = outputs_root()/ "metrics"/ "s2t_pr_sweep.csv"
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    print("Saved:", out.resolve())

if __name__=="__main__":
    main()
