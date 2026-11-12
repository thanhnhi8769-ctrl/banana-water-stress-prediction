# Dự báo sớm water stress và quyết định tưới (chuối Williams, FAO-56)

Pipeline 2 tầng:
1. Dự báo mức stress tuần t+1 (Healthy < Mild < Moderate < Severe) từ dữ liệu cây + FAO-56.
2. Chuyển xác suất dự báo thành quyết định tưới theo chi phí kỳ vọng.

## Cài đặt

Python 3.14.8 (bản đã dùng để chạy ra kết quả). Các thư viện được ghim phiên bản trong `requirements.txt`.

**Windows (PowerShell)**
```powershell
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

**Linux / AWS EC2 (Ubuntu)**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
Nếu Python mặc định của máy thấp hơn 3.14, dùng `uv`:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Chạy

Dữ liệu **không nằm trong repo** (đã có trong `.gitignore`). Tự đặt file vào
`data/raw/dataset_operacional_final.csv`, rồi chạy từ thư mục gốc của repo.

**Windows**
```powershell
py src/00b_validate_data.py   # chạy TRƯỚC, rồi đọc outputs/validation_report.txt
py run_all.py                 # 01 -> 02 -> 03 -> 10 -> 11 -> 13
```

**Linux**
```bash
python3 src/00b_validate_data.py
python3 run_all.py
```

Chưa có dữ liệu thật: `py src/make_fake_data.py` tạo dữ liệu giả
(chỉ để thử code, không dùng cho kết luận).

`run_all.py` tự tạo thư mục `outputs/` và `outputs/figures/`.

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
| `run_all.py` | Chạy lần lượt 01 -> 02 -> 03 -> 10 -> 11 -> 13 |

## Quy ước chống rò rỉ

- Chia theo cây (`PlantUID`), test = 20% cây phân tầng theo treatment.
- Loại `soil_moisture`, `VWC`, `riego_mm`, `estres_*`, `f_depl`, `s_norm`, `trat` khỏi feature.
- Savitzky-Golay chỉ dùng các tuần <= t. Calibration fit bên trong CV.
- Bộ `*_noWD` bỏ `water_deficit` để kiểm tra biến này có mang thông tin treatment không.
- Mốc đối chứng bắt buộc: Most Frequent và Persistence (t+1 = mức stress hiện tại).

## Kết quả (dữ liệu thật: 130 cây, 26 tuần, test 26 cây)

Cấu hình được chọn: bộ feature `D_full`, model `Ordinal_LogReg`, 24 biến.

| | F1 | QWK | MAE |
|---|---|---|---|
| Mô hình (test) | 0.864 | 0.935 | 0.136 |
| Persistence (t+1 = hiện tại) | 0.837 | 0.923 | 0.163 |

- Khoảng tin cậy 95% (bootstrap theo cây): QWK [0.900, 0.957], F1 [0.811, 0.892].
- Mô hình chỉ nhỉnh hơn Persistence một chút, và khoảng tin cậy của QWK chứa mức 0.923.
  Chưa thể kết luận mô hình vượt trội.
- Calibration không cải thiện Brier (0.1924 trước và sau), nên dùng xác suất thô.
- Quyết định tưới (tỷ lệ chi phí 5:1): chính sách Expected-cost dùng khoảng 47% lượng nước
  so với luôn tưới 100% ETc, chi phí kỳ vọng 0.152 so với 0.640.
- Chỉ có 26 cây test nên các con số còn dao động khá nhiều.

## Hạn chế và việc cần kiểm tra trước khi kết luận

1. **`water_deficit` có tính từ lượng tưới / treatment không?** Khi bỏ biến này
   (bộ `B_noWD`), QWK giảm còn khoảng 0.2, nên mô hình phụ thuộc rất nhiều vào nó.
   Nếu nó suy ra từ lượng tưới thì đây là rò rỉ gián tiếp.
2. **`estres_*` được tạo thế nào** (độc lập với feature hay suy ra từ feature)?
   Kiểm tra qua mô tả dataset hoặc hỏi nguồn cung cấp dữ liệu.
3. **Cost matrix**: nhu cầu nước theo mức stress (`NEED` trong `11_final.py`) và
   tỷ lệ chi phí 5:1 đến 50:1 là **giả định**, chưa có số liệu thực tế xác nhận.
4. **Giấy phép dữ liệu**: kiểm tra nguồn và điều khoản sử dụng của dataset trước khi
   đưa lên cloud hoặc công bố kết quả. Không đưa dữ liệu thật lên GitHub công khai.

## Chạy trên AWS (tùy chọn)

Một máy EC2 nhỏ (Ubuntu 24.04, 4 vCPU, không cần GPU) là đủ.

```bash
sudo apt update && sudo apt install -y git tmux awscli
git clone https://github.com/thanhnhi8769-ctrl/banana-water-stress-prediction.git
cd banana-water-stress-prediction
# cài môi trường như mục "Cài đặt" (Linux)
mkdir -p data/raw
aws s3 cp s3://TEN-BUCKET/dataset_operacional_final.csv data/raw/
tmux new -s train
python3 src/00b_validate_data.py
python3 run_all.py 2>&1 | tee outputs/run.log
aws s3 sync outputs/ s3://TEN-BUCKET/outputs/
```
Chạy xong nhớ **Stop hoặc Terminate** instance để không bị tính tiền.
