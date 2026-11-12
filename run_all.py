"""Chạy toàn bộ pipeline: py run_all.py   (chạy từ thư mục nckh)"""
import os
import subprocess
import sys

os.makedirs("outputs/figures", exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

STEPS = ["01_data_cleaning.py", "02_signal_denoising.py", "03_feature_engineering.py",
         "10_ablation_study.py", "11_final.py", "13_plots.py"]
env = dict(os.environ, PYTHONUTF8="1")
for s in STEPS:
    print(f"\n===== {s} =====", flush=True)
    if subprocess.run([sys.executable, f"src/{s}"], env=env).returncode:
        sys.exit(f"Dừng vì {s} bị lỗi")
print("\nXong. Kết quả ở thư mục outputs/")