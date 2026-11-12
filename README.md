# Dự báo sớm water stress và quyết định tưới (chuối Williams, FAO-56)

Pipeline 2 tầng: (1) dự báo mức stress tuần t+1 (Healthy < Mild < Moderate < Severe) từ dữ liệu cây + FAO-56;
(2) chuyển xác suất đã calibration thành quyết định tưới theo chi phí kỳ vọng.

## Cài đặt (Windows)
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Chạy
Đặt dữ liệu vào `data/raw/dataset_operacional_final.csv`, rồi:
```powershell
py src/00b_validate_data.py   # kiểm tra nhãn / rò rỉ (chạy TRƯỚC, đọc outputs/validation_report.txt)
py run_all.py                 # 01 -> 02 -> 03 -> 10 -> 11 -> 13
```
Chưa có dữ liệu thật: `py src/make_fake_data.py` tạo dữ liệu giả (chỉ để thử code, không dùng cho kết luận).

## Cấu trúc
| File | Việc |
|---|---|
| `src/config.py` | Tên cột, cột nhãn (sửa khi dùng dữ liệu thật) |
| `src/00b_validate_data.py` | Kiểm tra cấu trúc, nhãn có phải treatment đổi tên không, rò rỉ, `water_deficit` |
| `src/01_data_cleaning.py` | Chuẩn hóa nhãn 0..3, bỏ cột leak, điền thiếu theo cây |
| `src/02_signal_denoising.py` | Savitzky-Golay nhân quả, Z-score theo tuần, `weight_stale` |
| `src/03_feature_engineering.py` | Δ, lag, tương tác vật lý, nhãn t+1 |
| `src/common.py` | Bộ feature, chia theo cây, model, Frank-Hall ordinal, metric |
| `src/10_ablation_study.py` | 7 bộ feature x 6 model, CV 5 fold theo cây |
| `src/11_final.py` | Calibration, test 1 lần, bootstrap, quyết định tưới |
| `src/13_plots.py` | 6 biểu đồ trong `outputs/figures/` |

## Quy ước chống rò rỉ
- Chia theo cây (`PlantUID`), test = 20% cây phân tầng theo treatment.
- Loại `soil_moisture`, `VWC`, `riego_mm`, `estres_*`, `f_depl`, `s_norm`, `trat` khỏi feature.
- Savitzky-Golay chỉ dùng các tuần <= t. Calibration fit bên trong CV.
- Bộ `*_noWD` bỏ `water_deficit` để kiểm tra biến này có mang thông tin treatment không.
- Mốc đối chứng bắt buộc: Most Frequent và Persistence (t+1 = mức stress hiện tại).

## Cần xác nhận với giảng viên trước khi kết luận
1. `estres_*` được tạo thế nào (độc lập hay suy từ feature)?
2. `water_deficit` có tính từ lượng tưới không?
3. Cost matrix: nhu cầu nước theo mức stress (`NEED` trong `11_final.py`) và tỷ lệ chi phí 5:1 đến 50:1.
