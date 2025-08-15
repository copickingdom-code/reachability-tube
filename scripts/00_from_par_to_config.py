# scripts/00_from_par_to_config.py
import argparse, yaml, os, sys
from pathlib import Path

# ensure project root on sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from vehicletube.par_parse import derive_params

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--par", required=True, help="Path to CarSim .par file")
    ap.add_argument("--out", default="configs/default.yaml", help="Output YAML")
    ap.add_argument("--merge", action="store_true", help="Merge into existing YAML; only update 'vehicle'")
    args = ap.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    # 1) 读旧配置（若存在）
    base = {}
    if args.merge and out.exists():
        with open(out, "r", encoding="utf-8") as f:
            base = yaml.safe_load(f) or {}

    # 2) 解析 par，写入 vehicle（只更新 vehicle 子键）
    veh = derive_params(args.par)
    base.setdefault("vehicle", {}).update(veh)

    # 3) 如果没有设置必要默认值，给出兜底（不会覆盖已有值）
    base.setdefault("horizon_s", 2.0)
    base.setdefault("dt_s", 0.01)
    base.setdefault("mus", [0.3, 0.5, 0.8])
    base.setdefault("v0_kph", [30, 60, 120])
    base.setdefault("tube", {}).setdefault("erosion_m", 0.10)
    base.setdefault("s2t", {})
    base["s2t"].setdefault("area_eps_m2", 0.20)
    base["s2t"].setdefault("frac_thr", 0.90)
    base["s2t"].setdefault("roi_radius_m", 15.0)

    with open(out, "w", encoding="utf-8") as f:
        yaml.safe_dump(base, f, allow_unicode=True, sort_keys=False)
    print(f"[00] wrote {out}")

if __name__ == "__main__":
    main()
