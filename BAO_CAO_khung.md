# Dự báo sớm water stress và quyết định tưới cho cây chuối bằng đặc trưng vật lý FAO-56

## 1. Mở đầu
- Bối cảnh: tưới thiếu/thừa, nhu cầu dự báo sớm. (điền)
- RQ1: Đặc trưng FAO-56 có cải thiện dự báo water stress tuần t+1 không?
- RQ2: Xác suất đã calibration chuyển thành quyết định tưới theo rủi ro-chi phí thế nào?
- Đóng góp: (điền 2-3 ý)

## 2. Dữ liệu
- Chuối Musa AAA cv. Williams, Zona Bananera (Colombia), 2024-04-01 đến 2024-09-23.
- 130 cây x 26 tuần; 4 nhóm T1 30%, T2 50%, T3 75%, T4 100% ETc; thời tiết NASA POWER; ET0 Penman-Monteith.
- Nguồn gốc nhãn `estres_*`: (điền sau khi xác nhận) | Kết quả kiểm tra: `outputs/validation_report.txt`.

## 3. Phương pháp
3.1 Tiền xử lý: Savitzky-Golay nhân quả (cửa sổ 5, bậc 2), Z-score theo tuần, LOCF + `days_since_*`, `weight_stale`.
3.2 Đặc trưng: A (cây), B (FAO + môi trường), C = A+B, D = C + tương tác vật lý (VPD_per_LAI, Chlorophyll_Proxy, Deficit_per_LAI), lag, độ tươi dữ liệu.
3.3 Mô hình: Most Frequent, Logistic Regression, Random Forest, XGBoost, Ordinal (Frank-Hall) LogReg/XGB.
3.4 Đánh giá: tách 20% cây test (phân tầng theo treatment); CV 5 fold StratifiedGroupKFold trên 80% còn lại; Macro-F1, QWK, MAE, Brier; bootstrap theo cây; mốc Persistence.
3.5 Calibration: Platt scaling bên trong CV.
3.6 Quyết định: chi phí kỳ vọng với hành động {0, 50%, 100% ETc}; thiếu nước tốn R/đơn vị ETc, tưới thừa tốn 1/đơn vị; R = 5, 10, 20, 50.

## 4. Kết quả
| Hình/Bảng | Nguồn | Nội dung |
|---|---|---|
| Bảng 1 | `outputs/ablation_summary.csv` | Ablation A/B/C/D và các bộ `_noWD` |
| Hình 1 | `fig1_stress_va_ndvi_theo_tuan.png` | Stress, NDVI theo tuần |
| Hình 2 | `fig6_ablation.png` | QWK theo bộ feature |
| Bảng 2 | `11_final.py` | Test: F1, QWK, MAE, KTC 95%, so với Persistence |
| Hình 3 | `fig2_ma_tran_nham_lan.png` | Ma trận nhầm lẫn |
| Hình 4 | `fig3_calibration.png` | Độ tin cậy xác suất |
| Hình 5 | `fig5_do_quan_trong_bien.png` | Độ quan trọng biến |
| Bảng 3, Hình 6 | `outputs/decision.csv`, `fig4_quyet_dinh_tuoi.png` | Chi phí, thiếu nước, lượng nước theo R |

Trả lời RQ1: (điền: C so với A, D so với C, bỏ `water_deficit` thay đổi gì, có vượt Persistence không).
Trả lời RQ2: (điền: expected-cost so với argmax và luôn tưới 100%).

## 5. Thảo luận và hạn chế
- Chỉ 130 cây độc lập, 26 cây test: khoảng tin cậy rộng.
- Thời tiết giống nhau mọi cây trong tuần: chỉ có tác dụng khi kết hợp dữ liệu cây.
- Rủi ro nhãn/`water_deficit` phụ thuộc treatment (kết quả kiểm tra `_noWD`).
- Cost matrix là giả định, cần hiệu chỉnh với chuyên gia; 1 vụ, 1 địa điểm.

## 6. Hướng phát triển
Digital twin (Unity): PlantUID -> fila/columna -> heatmap xác suất stress -> mô phỏng tưới.

## Tài liệu tham khảo
FAO Irrigation and Drainage Paper 56 (Allen et al., 1998); AquaCrop (FAO); Frank & Hall (2001) ordinal classification; (thêm các bài về UAV NDVI/NDRE và water stress cây chuối).
