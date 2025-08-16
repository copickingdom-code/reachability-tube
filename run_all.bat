@echo off
setlocal EnableExtensions EnableDelayedExpansion

REM ===== Minimal, ASCII-only batch to run full pipeline =====
REM 1) EDIT THESE THREE LINES TO MATCH YOUR MACHINE
set "PAR=%CD%\data\vehicle\C-ClassHatchback.par"
@REM set "MUS=0.3 0.5 0.8"
@REM set "V0S=30 60 120"
REM If your raw CSVs are already under data\raw_runs\{mu}\{v0}\run_XXXX.csv just keep RAW_DIR as below
set "RAW_DIR=%CD%\data\raw_runs"

REM 2) Python and paths
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=UTF-8"
pushd "%~dp0"
set "ROOT=%CD%"
set "PY=python"
set "PYTHONPATH=%ROOT%;%PYTHONPATH%"

REM 3) Folders and log
for %%D in ("outputs" "outputs\manifests" "outputs\metrics" "outputs\figures" "outputs\gifs") do if not exist %%D mkdir %%D
for %%D in ("slices_overlay" "terminal_envelope" "area_vs_time" "tube3d" "s2t") do if not exist "outputs\figures\%%~D" mkdir "outputs\figures\%%~D"
set "LOG=outputs\run_all.log"
echo ===== VehicleTube Paper A pipeline ===== > "%LOG%"

echo [00] from_par_to_config
"%PY%" "scripts\00_from_par_to_config.py" --par "%PAR%" --out "configs\default.yaml" --merge >> "%LOG%" 2>&1

@REM echo [02] ingest_runs
@REM "%PY%" "scripts\02_ingest_runs.py" --mu %MUS% --v0 %V0S% --raw "%RAW_DIR%" --out "data\processed\raw_traj.h5" >> "%LOG%" 2>&1

echo [03] filter_stability
"%PY%" "scripts\03_filter_stability.py" >> "%LOG%" 2>&1

echo [04] build_tube_bins
"%PY%" "scripts\04_build_tube_bins.py" >> "%LOG%" 2>&1

echo [06a] area_vs_time
"%PY%" "scripts\06a_area_vs_time.py" >> "%LOG%" 2>&1

echo [06c] slice_overlays
"%PY%" "scripts\06c_draw_slice_overlays.py" >> "%LOG%" 2>&1

echo [06d] terminal_envelope
"%PY%" "scripts\06d_draw_terminal_envelope.py" >> "%LOG%" 2>&1

@REM echo [06e] mu_scaling_2d
@REM "%PY%" "scripts\06e_mu_scaling_2d.py" >> "%LOG%" 2>&1

echo [06f] tube_3d
"%PY%" "scripts\06f_plot_tube_3d.py" >> "%LOG%" 2>&1

echo [06g] tube_gif
"%PY%" "scripts\06g_make_tube_gif.py" >> "%LOG%" 2>&1

echo [06] build_figures
"%PY%" "scripts\06_figures.py" >> "%LOG%" 2>&1

echo [06] build_LTV
"%PY%" "scripts\06b_identify_A_LTV.py" >> "%LOG%" 2>&1

echo [07a] s2t_eval_geometry
"%PY%" "scripts\07a_s2t_eval_geometry.py" >> "%LOG%" 2>&1

echo [07b] s2t_eval_with_runs
"%PY%" "scripts\07b_s2t_eval_with_runs.py" >> "%LOG%" 2>&1

echo [07c] sweep_thresholds
"%PY%" "scripts\07c_sweep_thresholds.py" >> "%LOG%" 2>&1

echo [07d] group_report
"%PY%" "scripts\07d_group_report.py" >> "%LOG%" 2>&1

echo [07e] list_fp_fn
"%PY%" "scripts\07e_list_fp_fn.py" >> "%LOG%" 2>&1

echo [07f] plot_s2t_sweep
"%PY%" "scripts\07f_plot_s2t_sweep.py" >> "%LOG%" 2>&1

echo [99] pack outputs
"%PY%" "scripts\99_pack_outputs.py" >> "%LOG%" 2>&1

echo [06e] mu_scaling_4d
"%PY%" "scripts\06e_mu_scaling_4d.py" >> "%LOG%" 2>&1

echo [04b] 04b_volume4d_timeseries
"%PY%" "scripts\04b_volume4d_timeseries.py" >> "%LOG%" 2>&1

echo [baselines] pilot_baselines
"%PY%" "scripts\baselines\eval_pilot_baselines.py" >> "%LOG%" 2>&1

echo [closed loop] closed_loop
"%PY%" "scripts/closedloop/aeb_gate_eval.py" >> "%LOG%" 2>&1

echo [analysis] monotonicity_timeseries
"%PY%" "scripts/analysis/monotonicity_timeseries.py" >> "%LOG%" 2>&1

echo Done. Outputs under .\outputs\
echo (Log: %LOG%)

endlocal
