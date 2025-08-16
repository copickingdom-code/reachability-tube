"# Reachability Tube (Finite-Time)" 
# Vehicle‑Driven Long‑Tail Safety via Finite‑Time Reachability Tubes

> 复现论文主要结果（Tube 构建、收敛性、摩擦缩放、极端场景评测）并给出可复用的工程脚本与图表导出。
- 论文：**Vehicle‑Driven Long‑Tail Safety via Finite‑Time Reachability Tubes**（详述方法、指标与表图编号，见本仓库 `paper/` 或预印本）。
- 该方法以**有限时间可达性管**（finite‑time reachability tube, 记作 `𝒯(t,e)`）为核心，离线采样两段式控制、用解析稳定性筛选（侧偏角、横摆角速度、载荷转移、纵向附着、线性化特征值）
- 以 **α‑shape** 拟合每个时间切片的可达集合；随后给出体积/豪斯多夫距离的收敛性与摩擦缩放关系，并用几种规则对极端场景进行几何判定评测。:contentReference[oaicite:1]{index=1}

---

## TL;DR

- `scripts/build_tube.py`：生成按时间切片的可达性管（4D 特征，默认 `Δx, Δy, v, ψ`），并保存关键帧与几何数据。
- `scripts/metrics_convergence.py`：计算时间积分 4D 体积、相邻切片 Hausdorff 漂移等收敛性指标。
- `scripts/fit_mu_scaling.py`：拟合摩擦缩放律 `Vol ≈ C μ^p` 并输出拟合图（Fig 2）。
- `scripts/eval_s2t.py`：对 216 个极端场景做 Tube 几何判定评估（Rule A / Rule B），导出混淆矩阵和分组表。
- `scripts/plot_figures.py`：批量导出主要图（如每 μ 的切片叠加图、收敛曲线、PR/阈值 sweep 等）。

---

## 环境安装

建议 Python 3.10+

# 安装依赖
pip install -r requirements.txt
