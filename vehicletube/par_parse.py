
import re, math, yaml, numpy as np, pathlib as _pl

def _find(txt, key):
    m = re.search(r'\b'+re.escape(key)+r'\s+([\-0-9\.]+)', txt)
    return float(m.group(1)) if m else None

def parse_vehicle_from_par(par_path:str)->dict:
    txt = _pl.Path(par_path).read_text(encoding='utf-8', errors='ignore')
    vals = {}
    vals['LX_AXLE_mm'] = _find(txt, 'LX_AXLE')
    vals['LX_CG_SU_mm'] = _find(txt, 'LX_CG_SU')
    vals['M_SU_kg'] = _find(txt, 'M_SU')
    vals['IZZ_SU'] = _find(txt, 'IZZ_SU')
    vals['H_CG_SU_mm'] = _find(txt, 'H_CG_SU')
    m = re.search(r'Front Track\s*=\s*([0-9\.]+)\s*mm', txt)
    if m: vals['FrontTrack_mm'] = float(m.group(1))
    occ = list(re.finditer(r'\n\s*M_US\s+([0-9\.]+)', txt))
    if len(occ)>=1: vals['M_US_front'] = float(occ[0].group(1))
    if len(occ)>=2: vals['M_US_rear'] = float(occ[1].group(1))
    mfy = re.search(r'/F Tires\\Fy\\TireFy_.*?PARSFILE', txt, flags=re.S)
    fy_block = None
    if mfy:
        block = mfy.group(0)
        mdat = re.search(r'\*3D_DATA\s+\d+,\s*\d+[^\n]*\n(FY_TIRE_CARPET.*)', block, flags=re.S)
        if mdat: fy_block = mdat.group(1)
    vals['tire_fy_block'] = fy_block
    vals['FZ_REF'] = _find(txt, 'FZ_REF')
    return vals

def _parse_fy_carpet(fy_block:str):
    lines = [ln.strip() for ln in fy_block.splitlines() if ',' in ln]
    header = [float(x.strip()) for x in lines[0].split(',')]
    fz_cols = np.array(header[1:], dtype=float)
    rows = []
    for ln in lines[1:]:
        parts = [p.strip() for p in ln.split(',')]
        try: alpha = float(parts[0])
        except: continue
        vals = []
        for p in parts[1:]:
            try: vals.append(float(p))
            except: vals.append(np.nan)
        if len(vals)==len(fz_cols):
            rows.append((alpha, vals))
    alphas = np.array([r[0] for r in rows], dtype=float)
    arr = np.array([r[1] for r in rows], dtype=float)
    return fz_cols, alphas, arr

def _slope_at_origin(fz_cols, alphas, arr, fz_query):
    def idx_of(val):
        idx = np.where(np.isclose(alphas, val))[0]
        return idx[0] if len(idx)>0 else None
    i05 = idx_of(0.5); i10 = idx_of(1.0)
    if i05 is None or i10 is None:
        raise RuntimeError('TireFy table must contain 0.5 and 1.0 deg rows')
    dFy = arr[i10,:] - arr[i05,:]
    d_alpha = np.deg2rad(0.5)
    slope = dFy / d_alpha
    if fz_query <= fz_cols.min(): return float(slope[0])
    if fz_query >= fz_cols.max(): return float(slope[-1])
    j = np.searchsorted(fz_cols, fz_query)
    fz0,fz1 = fz_cols[j-1], fz_cols[j]
    s0,s1 = slope[j-1], slope[j]
    return float(s0 + (s1-s0)*(fz_query-fz0)/(fz1-fz0))

def derive_params(par_path:str)->dict:
    import numpy as np
    info = parse_vehicle_from_par(par_path)
    for k in ['LX_AXLE_mm','LX_CG_SU_mm','M_SU_kg','IZZ_SU','H_CG_SU_mm']:
        if info.get(k) is None: raise RuntimeError('Missing '+k+' in .par')

    L = info['LX_AXLE_mm']/1000.0
    lf = info['LX_CG_SU_mm']/1000.0
    lr = L - lf
    t_w = (info.get('FrontTrack_mm', 1540.0))/1000.0
    h_cg = info['H_CG_SU_mm']/1000.0
    m_su = info['M_SU_kg']
    m_us_f = info.get('M_US_front', 0.0)
    m_us_r = info.get('M_US_rear', 0.0)
    m_total = m_su + m_us_f + m_us_r
    g = 9.81
    Iz = info['IZZ_SU'] + 0.5*(m_us_f+m_us_r)*(t_w/2.0)**2

    Cf0 = 1.2e5; Cr0 = 1.0e5
    if info['tire_fy_block']:
        fz_cols, alphas, arr = _parse_fy_carpet(info['tire_fy_block'])
        F_front_axle = m_total*g * (lr/L)
        F_rear_axle  = m_total*g * (lf/L)
        Fz_front_wheel = F_front_axle/2.0
        Fz_rear_wheel  = F_rear_axle/2.0
        # compute slopes
        dFy = arr[np.where(np.isclose(alphas,1.0))[0][0],:] - arr[np.where(np.isclose(alphas,0.5))[0][0],:]
        d_alpha = np.deg2rad(0.5)
        slope = dFy / d_alpha
        # linear interp
        def interp(fz_query):
            if fz_query <= fz_cols.min(): return float(slope[0])
            if fz_query >= fz_cols.max(): return float(slope[-1])
            j = np.searchsorted(fz_cols, fz_query)
            fz0,fz1 = fz_cols[j-1], fz_cols[j]
            s0,s1 = slope[j-1], slope[j]
            return float(s0 + (s1-s0)*(fz_query-fz0)/(fz1-fz0))
        Cf0 = 2.0 * interp(Fz_front_wheel)
        Cr0 = 2.0 * interp(Fz_rear_wheel)

    return dict(
        m_kg=round(m_total,3),
        Iz_kgm2=round(Iz,3),
        lf_m=round(lf,4),
        lr_m=round(lr,4),
        track_m=round(t_w,4),
        h_cg_m=round(h_cg,4),
        Cf0_Nprad_front=round(Cf0,1),
        Cr0_Nprad_rear=round(Cr0,1),
    )

def save_yaml(params:dict, out_yaml:str):
    with open(out_yaml,'w',encoding='utf-8') as f:
        yaml.safe_dump(dict(vehicle=params), f, sort_keys=False, allow_unicode=True)

if __name__=='__main__':
    import sys, os
    if len(sys.argv)<2:
        print('Usage: python vehicletube/par_parse.py path\to\vehicle.par [configs\vehicle_from_par.yaml]')
        sys.exit(1)
    par = sys.argv[1]
    out = sys.argv[2] if len(sys.argv)>=3 else 'configs/vehicle_from_par.yaml'
    params = derive_params(par)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    save_yaml(params, out)
    print('Wrote', out)
