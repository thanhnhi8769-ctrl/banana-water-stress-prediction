# Dự báo sớm water stress và quyết định tưới (chuối Williams, FAO-56)

Pipeline 2 tầng:
1. Dự báo mức stress tuần t+1 (Healthy < Mild < Moderate < Severe) từ dữ liệu cây + FAO-56.
2. Chuyển xác suất dự báo thành quyết định tưới theo chi phí kỳ vọng.

## Cài đặt

Python 3.14.8 (bản đã dùng để chạy thử code). Các thư viện được ghim phiên bản trong `requirements.txt`.

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
| `src/make_fake_data.py` | Tạo dữ liệu giả để thử code |
| `run_all.py` | Chạy lần lượt 01 -> 02 -> 03 -> 10 -> 11 -> 13 |

## Quy ước chống rò rỉ

- Chia theo cây (`PlantUID`), test = 20% cây phân tầng theo treatment.
- Loại `soil_moisture`, `VWC`, `riego_mm`, `estres_*`, `f_depl`, `s_norm`, `trat` khỏi feature.
- Savitzky-Golay chỉ dùng các tuần <= t. Calibration fit bên trong CV.
- Bộ `*_noWD` bỏ `water_deficit` để kiểm tra biến này có mang thông tin treatment không.
- Mốc đối chứng bắt buộc: Most Frequent và Persistence (t+1 = mức stress hiện tại).

## Kết quả

Kết quả chính thức **chưa có**: các lần chạy đến nay dùng dữ liệu giả
(`make_fake_data.py`), chỉ để kiểm tra code, không dùng cho kết luận.
Sau khi chạy với dữ liệu thật, điền số liệu vào đây.

## Hạn chế và việc cần kiểm tra trước khi kết luận

1. **`water_deficit` có tính từ lượng tưới / treatment không?** Nếu có thì đây là rò rỉ
   gián tiếp. Chạy `00b_validate_data.py` và bộ `*_noWD` trên dữ liệu thật để kiểm tra.
2. **Nhãn `estres_*` phải đo độc lập với feature.** Khi xây dataset, nên lấy nhãn từ đo
   sinh lý (thế nước lá, độ dẫn khí khổng) hoặc đánh giá chuyên gia, không tính từ
   NDVI, `water_deficit` hay treatment. Ghi rõ cách tạo nhãn trong báo cáo.
3. **Cost matrix**: nhu cầu nước theo mức stress (`NEED` trong `11_final.py`) và
   tỷ lệ chi phí 5:1 đến 50:1 là **giả định**, chưa có số liệu thực tế xác nhận.
4. **Quyền sử dụng dữ liệu**: nếu dataset có phần lấy từ nguồn khác (ảnh vệ tinh,
   trạm thời tiết, nông trại), kiểm tra điều khoản trước khi đưa lên cloud hoặc công bố.
   Không đưa dữ liệu thật lên GitHub công khai.

## Chạy trên AWS (đã thử)

Đã chạy thành công trên EC2 `t3.micro`, Ubuntu 24.04, bằng dữ liệu giả.
Kết quả trùng khớp với máy cá nhân, nên môi trường tái lập được.
Chưa thử với dữ liệu thật.

Lưu ý khi tạo máy:
- Security group: SSH (port 22), nguồn **My IP** dạng IPv4. Nếu SSH báo `Connection timed out`,
  kiểm tra rule không bị gắn với địa chỉ IPv6.
- Máy `t3.micro` chỉ có 1 GB RAM nên tạo swap 2 GB trước khi chạy.

```bash
sudo fallocate -l 2G /swapfile && sudo chmod 600 /swapfile
sudo mkswap /swapfile && sudo swapon /swapfile

sudo apt update && sudo apt install -y git tmux
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

git clone https://github.com/thanhnhi8769-ctrl/banana-water-stress-prediction.git
cd banana-water-stress-prediction
uv venv --python 3.14 .venv
source .venv/bin/activate
uv pip install -r requirements.txt

tmux new -s train
python src/make_fake_data.py        # hoặc chép dữ liệu thật vào data/raw/ bằng scp
python src/00b_validate_data.py
python run_all.py 2>&1 | tee run.log
```

Tải kết quả về (PowerShell trên máy cá nhân):
```powershell
scp -i "banana.pem" -r ubuntu@<public-ip>:~/banana-water-stress-prediction/outputs .\outputs_aws
```

Xong thì **Terminate** instance.
