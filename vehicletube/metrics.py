
import pandas as pd
import shapely.wkt as wkt
import shapely.ops as ops

def layered_areas_to_frame(layered):
    rows = []
    for entry in layered:
        k = entry['time_index']
        for b in entry['bins']:
            rows.append(dict(time_index=k, v_min=b['v_bin'][0], v_max=b['v_bin'][1],
                             psi_min=b['psi_bin'][0], psi_max=b['psi_bin'][1], area=b['area']))
    return pd.DataFrame(rows)

def union_area_per_time(layered):
    out = []
    for entry in layered:
        polys = [wkt.loads(b['wkt']) for b in entry['bins'] if 'wkt' in b]
        if not polys:
            out.append(dict(time_index=entry['time_index'], area=0.0)); continue
        union = ops.unary_union(polys)
        out.append(dict(time_index=entry['time_index'], area=float(union.area)))
    return pd.DataFrame(out)
