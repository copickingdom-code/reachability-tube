# -*- coding: utf-8 -*-
"""
99_pack_outputs — 规范化 outputs 目录结构
"""
import shutil, glob, os
from pathlib import Path

ROOT = Path(".")
OUT = ROOT / "outputs"


def safe_mkdir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def mv(pattern, dst_dir):
    files = [Path(p) for p in glob.glob(pattern)]
    if not files:
        return 0
    safe_mkdir(dst_dir)
    n = 0
    for f in files:
        try:
            target = dst_dir / f.name
            if target.resolve() == f.resolve():
                continue
            if target.exists():
                # 若重名，改个后缀
                stem = target.stem;
                suf = target.suffix
                k = 1
                while True:
                    alt = target.with_name(f"{stem}__{k}{suf}")
                    if not alt.exists():
                        target = alt;
                        break
                    k += 1
            shutil.move(str(f), str(target))
            n += 1
        except Exception as e:
            print("[WARN] move fail:", f, "→", e)
    return n


def main():
    # 目标目录
    figs_tubes_area = OUT / "figures" / "tubes" / "area_vs_time"
    figs_tubes_slices = OUT / "figures" / "tubes" / "slices"
    figs_tubes_term = OUT / "figures" / "tubes" / "terminal"
    figs_tubes_3d = OUT / "figures" / "tubes" / "tube3d"
    figs_tubes_gif = OUT / "figures" / "tubes" / "gifs"
    figs_s2t_over = OUT / "figures" / "s2t" / "overlays"
    figs_s2t_pr = OUT / "figures" / "s2t" / "pr_sweep"
    figs_margin_ident_pr = OUT / "figures" / "margin_ident"
    figs_mu_scaling_pr = OUT / "figures" / "mu_scaling"

    mets_tubes = OUT / "metrics" / "tubes"
    mets_s2t = OUT / "metrics" / "s2t"

    safe_mkdir(figs_tubes_area)
    safe_mkdir(figs_tubes_slices)
    safe_mkdir(figs_tubes_term)
    safe_mkdir(figs_tubes_3d)
    safe_mkdir(figs_tubes_gif)
    safe_mkdir(figs_s2t_over)
    safe_mkdir(figs_s2t_pr)
    safe_mkdir(mets_tubes)
    safe_mkdir(mets_s2t)
    safe_mkdir(figs_margin_ident_pr)
    safe_mkdir(figs_mu_scaling_pr)
    safe_mkdir(OUT / "logs")


    moved = 0
    # 图形：area_vs_time*
    moved += mv(str(OUT / "figures" / "area_vs_time_*.png"), figs_tubes_area)
    # 图形：slices_overlay_*
    moved += mv(str(OUT / "figures" / "slices_overlay_*.png"), figs_tubes_slices)
    # 图形：terminal_envelope_*
    moved += mv(str(OUT / "figures" / "terminal_envelope_*.png"), figs_tubes_term)
    # 图形：tube3d_*.png
    moved += mv(str(OUT / "figures" / "tube3d_*.png"), figs_tubes_3d)
    # GIF：tube3d_*.gif（如果你事先放在 figures 根目录）
    moved += mv(str(OUT / "figures" / "tube3d_*.gif"), figs_tubes_gif)
    # S2T 叠图（07a 输出）
    moved += mv(str(OUT / "figures" / "s2t" / "cutin_*.png"), figs_s2t_over)
    moved += mv(str(OUT / "figures" / "s2t" / "lead_*.png"), figs_s2t_over)
    moved += mv(str(OUT / "figures" / "s2t" / "ped_*.png"), figs_s2t_over)
    # PR 曲线（07f输出）
    moved += mv(str(OUT / "figures" / "s2t" / "s2t_*.png"), figs_s2t_pr)
    # margin_ident曲线
    moved += mv(str(OUT / "figures" / "spec_margin_ident*.png"), figs_margin_ident_pr)
    # mu_scaling曲线
    moved += mv(str(OUT / "figures" / "mu_scaling*.png"), figs_mu_scaling_pr)

    # 指标：tubes
    moved += mv(str(OUT / "metrics" / "area_union_mu*_v*.csv"), mets_tubes)
    moved += mv(str(OUT / "metrics" / "tube_volume_*.csv"), mets_tubes)
    moved += mv(str(OUT / "metrics" / "mu_scaling.csv"), mets_tubes)

    # 指标：s2t
    moved += mv(str(OUT / "metrics" / "s2t_eval_geometry.csv"), mets_s2t)
    moved += mv(str(OUT / "metrics" / "s2t_eval_with_runs.csv"), mets_s2t)
    moved += mv(str(OUT / "metrics" / "s2t_confusion_matrix.csv"), mets_s2t)
    moved += mv(str(OUT / "metrics" / "s2t_pr_sweep.csv"), mets_s2t)
    moved += mv(str(OUT / "metrics" / "s2t_group_report.csv"), mets_s2t)

    # manifests 保持原地；如需归档，可在此复制到 outputs\manifests_archive\

    print(f"[PACK] moved {moved} files into organized folders under outputs\\")


if __name__ == "__main__":
    main()
