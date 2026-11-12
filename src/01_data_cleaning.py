import pandas as pd
from pandas.api.types import is_numeric_dtype
from config import RAW_PATH, ENCODING, RENAME, LABEL_COL, LABEL_ORDER, REQUIRED

df = pd.read_csv(RAW_PATH, encoding=ENCODING).rename(columns=RENAME)

missing = [c for c in REQUIRED + [LABEL_COL] if c not in df.columns]
if missing:
    raise SystemExit(f"Thiếu cột: {missing}\nSửa RENAME trong src/config.py. Các cột hiện có:\n{list(df.columns)}")

# ---- chuẩn hóa nhãn về 0..3 (Healthy < Mild < Moderate < Severe) ----
y = df[LABEL_COL]
if is_numeric_dtype(y):
    vals = set(y.dropna().unique())
    if vals == {1, 2, 3, 4}:
        y = y - 1
    if not set(y.dropna().unique()) <= {0, 1, 2, 3}:
        raise SystemExit(f"Nhãn số phải thuộc 0..3, đang có: {sorted(vals)}")
    df["estres_nivel"] = y.astype(int)
else:
    order = {s.lower(): i for i, s in enumerate(LABEL_ORDER)}
    m = y.astype(str).str.strip().str.lower().map(order)
    if m.isna().any():
        raise SystemExit(f"Nhãn lạ: {sorted(y[m.isna()].astype(str).unique())}. Sửa LABEL_ORDER trong config.py")
    df["estres_nivel"] = m.astype(int)

# ---- chỉ giữ cột cần thiết (tự loại mọi cột leak như soil_moisture, riego_mm, f_depl, s_norm...) ----
keep = REQUIRED + [c for c in ("fila", "columna") if c in df.columns] + ["estres_nivel"]
df = df[keep].sort_values(["PlantUID", "semana"]).reset_index(drop=True)

# ---- điền thiếu theo từng cây (LOCF), không dùng thông tin cây khác ----
fill = ["NDVI_mean", "NDRE_mean", "LAI_locf", "altura_locf", "hojas_locf", "days_since_morfo", "days_since_lai"]
for c in fill:
    df[c] = df.groupby("PlantUID")[c].transform(lambda s: s.ffill().bfill())

n0 = len(df)
df = df.dropna().reset_index(drop=True)
print(f"Bỏ {n0 - len(df)} dòng còn thiếu | Còn thiếu: {int(df.isna().sum().sum())}")
print("Số cây:", df["PlantUID"].nunique(), "| số tuần:", df["semana"].nunique())
print(df.shape)
df.to_csv("data/clean.csv", index=False)
