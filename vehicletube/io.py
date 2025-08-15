
import pandas as pd, numpy as np

def read_run_csv(path, horizon_s=2.0, dt=0.01):
    df = pd.read_csv(path)
    tcol = 'Time' if 'Time' in df.columns else 'T_Stamp' if 'T_Stamp' in df.columns else None
    if tcol is None:
        N = int(horizon_s/dt)
        df = df.iloc[:N].copy()
        df['Time'] = (np.arange(len(df))*dt)
        tcol = 'Time'
    else:
        t0 = float(df[tcol].iloc[0])
        df = df[(df[tcol]-t0) < horizon_s].copy()
    def get(c): return df[c].to_numpy() if c in df.columns else None
    Vx_kph = get('Vx_SM'); Vy_kph = get('Vy_SM')
    if Vx_kph is not None and Vy_kph is not None:
        v_mps = (Vx_kph**2 + Vy_kph**2)**0.5/3.6
    else:
        v_mps = get('Speed (m/s)')
        if v_mps is None: raise RuntimeError('Missing Vx_SM/Vy_SM or Speed (m/s) in '+path)
    yaw_deg = get('Yaw') if get('Yaw') is not None else get('Yaw (deg)')
    yaw_rad = np.deg2rad(yaw_deg) if yaw_deg is not None else np.zeros_like(v_mps)
    r_deg = get('AVz') if get('AVz') is not None else get('Yaw Rate (deg/s)')
    r_rps = np.deg2rad(r_deg) if r_deg is not None else np.zeros_like(v_mps)
    beta_deg = get('Beta')
    beta_rad = np.deg2rad(beta_deg) if beta_deg is not None else np.zeros_like(v_mps)
    ax = get('Ax_SM') if get('Ax_SM') is not None else get('Long Accel (m/s^2)')
    ay = get('Ay_SM') if get('Ay_SM') is not None else get('Lat Accel (m/s^2)')
    out = dict(
        t = df[tcol].to_numpy(),
        x = get('Xcg_SM'), y = get('Ycg_SM'),
        v_mps = v_mps, yaw_rad = yaw_rad, r_rps = r_rps, beta_rad = beta_rad,
        ax = ax if ax is not None else np.zeros_like(v_mps),
        ay = ay if ay is not None else np.zeros_like(v_mps),
        Fz_L1 = get('Fz_L1'), Fz_L2 = get('Fz_L2'), Fz_R1 = get('Fz_R1'), Fz_R2 = get('Fz_R2'),
    )
    return out
