# scripts/04b_volume4d_timeseries.py
# 从 outputs/manifests/tube_layered_{mu}_{v0}.json 读取逐时刻分层几何，
# 用每个 bin 的边界推导 dv/dpsi，计算每个时刻的 4D 体积：
#     Volume4D(t) = sum_bins area_xy * dv * dpsi
# 并输出：
#   - outputs/metrics/volume4d_timeseries_mu{mu}_v{v0}.csv   (time_index, t_s, volume4d)
#   - outputs/metrics/volume4d_mu{mu}_v{v0}.csv              (mu, v0, dt, horizon, vol4d_time_integral)

from pathlib import Path
import os, json, glob, math
import pandas as pd

def try_import_outputs_root():
    try:
        from vehicletube.paths import outputs_root
        return outputs_root()
    except Exception:
        return Path("outputs").resolve()

OUT = Path(try_import_outputs_root())

def compute_volume4d_timeseries(manifest_path: Path):
    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    dt       = float(man.get("dt", 0.01))
    horizon  = float(man.get("horizon", dt * len(man.get("layered", []))))
    layered  = man["layered"]

    rows = []
    for layer in layered:
        k   = int(layer["time_index"])
        t_s = k * dt
        vol4d_k = 0.0
        for b in layer["bins"]:
            area = float(b.get("area", 0.0) or 0.0)

            vbin  = b.get("v_bin")   # [v_lo, v_hi]
            psibin= b.get("psi_bin") # [psi_lo, psi_hi]

            if isinstance(vbin, (list, tuple)) and len(vbin) == 2:
                dv = float(vbin[1]) - float(vbin[0])
            else:
                # 兜底：允许老版本 manifest 把步长放在顶层；如果也没有就报错
                dv = man.get("dv_mps", None)
                if dv is None:
                    raise ValueError(f"{manifest_path.name} 缺少 v_bin 或 dv_mps，无法计算 dv")

            if isinstance(psibin, (list, tuple)) and len(psibin) == 2:
                dpsi = float(psibin[1]) - float(psibin[0])
            else:
                dpsi = man.get("dpsi_rad", None)
                if dpsi is None:
                    raise ValueError(f"{manifest_path.name} 缺少 psi_bin 或 dpsi_rad，无法计算 dpsi")

            vol4d_k += area * dv * dpsi

        rows.append(dict(time_index=k, t_s=t_s, volume4d=vol4d_k))

    df = pd.DataFrame(rows).sort_values("time_index").reset_index(drop=True)
    vol4d_int = float((df["volume4d"] * dt).sum())
    meta = dict(mu=man.get("mu"), v0=man.get("v0"), dt=dt, horizon=horizon,
                vol4d_time_integral=vol4d_int)
    return df, meta

def main():
    man_dir = OUT / "manifests"
    met_dir = OUT / "metrics"
    fig_dir = OUT / "figures"
    met_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(glob.glob((man_dir / "tube_layered_*.json").as_posix()))
    if not files:
        print("[WARN] 没找到 manifests/tube_layered_*.json")
        return

    summary = []
    print("===== VehicleTube Paper A pipeline: 04b (4D volume from manifests) =====")
    for f in files:
        fpath = Path(f)
        try:
            df, meta = compute_volume4d_timeseries(fpath)
            mu = meta["mu"]; v0 = meta["v0"]

            # 逐时刻
            df_out = met_dir / f"volume4d_timeseries_mu{mu}_v{v0}.csv"
            df.to_csv(df_out, index=False, encoding="utf-8")

            # 单行汇总
            pd.DataFrame([meta]).to_csv(met_dir / f"volume4d_mu{mu}_v{v0}.csv",
                                        index=False, encoding="utf-8")
            summary.append(meta)
            print(f"[OK] {fpath.name} -> ∫Vol4D dt = {meta['vol4d_time_integral']:.3f}")
        except Exception as e:
            print(f"[ERR] {fpath.name}: {e}")

    if summary:
        pd.DataFrame(summary).sort_values(["v0","mu"]).to_csv(
            met_dir / "volume4d_summary.csv", index=False, encoding="utf-8")
        print(f"[DONE] 汇总写入 {met_dir/'volume4d_summary.csv'}")
    else:
        print("[WARN] 没有成功处理任何文件。")

if __name__ == "__main__":
    main()
