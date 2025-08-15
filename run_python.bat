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
set "LOG=outputs\run_python.log"
echo ===== VehicleTube Paper A pipeline ===== > "%LOG%"

@REM echo [00] from_par_to_config
@REM "%PY%" "scripts\00_from_par_to_config.py" --par "%PAR%" --out "configs\default.yaml" --merge >> "%LOG%" 2>&1

@REM echo [02] ingest_runs
@REM "%PY%" "scripts\02_ingest_runs.py" --mu %MUS% --v0 %V0S% --raw "%RAW_DIR%" --out "data\processed\raw_traj.h5" >> "%LOG%" 2>&1

@REM echo [03] filter_stability
@REM "%PY%" "scripts\03_filter_stability.py" >> "%LOG%" 2>&1
@REM
@REM echo [04] build_tube_bins
@REM "%PY%" "scripts\04_build_tube_bins.py" >> "%LOG%" 2>&1
@REM
@REM echo [06a] area_vs_time
@REM "%PY%" "scripts\06a_area_vs_time.py" >> "%LOG%" 2>&1
@REM
@REM echo [06c] slice_overlays
@REM "%PY%" "scripts\06c_draw_slice_overlays.py" >> "%LOG%" 2>&1
@REM
@REM echo [06d] terminal_envelope
@REM "%PY%" "scripts\06d_draw_terminal_envelope.py" >> "%LOG%" 2>&1
@REM
@REM echo [06e] mu_scaling
@REM "%PY%" "scripts\06e_mu_scaling.py" >> "%LOG%" 2>&1
@REM
@REM echo [06f] tube_3d
@REM "%PY%" "scripts\06f_plot_tube_3d.py" >> "%LOG%" 2>&1
@REM
@REM echo [06g] tube_gif
@REM "%PY%" "scripts\06g_make_tube_gif.py" >> "%LOG%" 2>&1

@REM echo [06] build_figures
@REM "%PY%" "scripts\06_figures.py" >> "%LOG%" 2>&1

echo [06h] build_convergence
"%PY%" "scripts\06h_convergence_metrics.py" >> "%LOG%" 2>&1

@REM echo [06] build_LTV
@REM "%PY%" "scripts\06b_identify_A_LTV.py" >> "%LOG%" 2>&1
@REM
@REM echo [07a] s2t_eval_geometry
@REM "%PY%" "scripts\07a_s2t_eval_geometry.py" >> "%LOG%" 2>&1
@REM
@REM echo [07b] s2t_eval_with_runs
@REM "%PY%" "scripts\07b_s2t_eval_with_runs.py" >> "%LOG%" 2>&1
@REM
@REM echo [07c] sweep_thresholds
@REM "%PY%" "scripts\07c_sweep_thresholds.py" >> "%LOG%" 2>&1
@REM
@REM echo [07d] group_report
@REM "%PY%" "scripts\07d_group_report.py" >> "%LOG%" 2>&1
@REM
@REM echo [07e] list_fp_fn
@REM "%PY%" "scripts\07e_list_fp_fn.py" >> "%LOG%" 2>&1
@REM
@REM echo [07f] plot_s2t_sweep
@REM "%PY%" "scripts\07f_plot_s2t_sweep.py" >> "%LOG%" 2>&1

@REM echo [99] pack outputs
@REM "%PY%" "scripts\99_pack_outputs.py" >> "%LOG%" 2>&1

echo Done. Outputs under .\outputs\
echo (Log: %LOG%)

endlocal
