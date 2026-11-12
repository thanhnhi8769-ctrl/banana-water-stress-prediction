# ===== CẤU HÌNH CHO DỮ LIỆU THẬT — chỉ cần sửa file này =====
RAW_PATH = "data/raw/dataset_operacional_final.csv"
ENCODING = "utf-8"
RENAME = {}  # đổi tên cột thật -> tên chuẩn, ví dụ {"NDVI": "NDVI_mean", "ETc_week": "ETc"}

# Cột nhãn stress. Sau khi chạy 00b_validate_data.py mới chọn cột đúng (độc lập, không suy từ feature).
LABEL_COL = "estres_nivel"
LABEL_ORDER = ["Healthy", "Mild", "Moderate", "Severe"]  # chỉ dùng nếu nhãn là chữ

# Tên chuẩn mà toàn bộ pipeline dùng
REQUIRED = ["PlantUID", "semana", "trat",
            "NDVI_mean", "NDRE_mean", "LAI_locf", "altura_locf", "hojas_locf",
            "days_since_morfo", "days_since_lai",
            "ET0_week_sum", "VPD_week_mean", "P_week_sum", "ETc", "water_deficit"]
