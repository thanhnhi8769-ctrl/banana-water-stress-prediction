import pandas as pd

df = pd.read_csv("data/denoised.csv").sort_values(["PlantUID", "semana"]).reset_index(drop=True)
g = df.groupby("PlantUID")

# Động lực học (chỉ dùng tuần t và trước đó)
df["dNDVI"] = g["NDVI_sg"].diff()
df["dNDRE"] = g["NDRE_sg"].diff()
df["Growth_Rate_Height"] = g["altura_locf"].diff() / (df["days_since_morfo"] + 1)
df["Growth_Rate_Leaves"] = g["hojas_locf"].diff() / (df["days_since_morfo"] + 1)
df["NDVI_lag1"] = g["NDVI_sg"].shift(1)
df["NDRE_lag1"] = g["NDRE_sg"].shift(1)

# Tương tác vật lý - sinh học
df["VPD_per_LAI"] = df["VPD_week_mean"] / (df["LAI_locf"] + 0.1)
df["Chlorophyll_Proxy"] = df["NDRE_sg"] * df["LAI_locf"]
df["Deficit_per_LAI"] = df["water_deficit"] / (df["LAI_locf"] + 0.1)

# Nhãn t+1
df["target_t1"] = g["estres_nivel"].shift(-1)

need = ["dNDVI", "dNDRE", "Growth_Rate_Height", "Growth_Rate_Leaves",
        "NDVI_lag1", "NDRE_lag1", "target_t1"]
df = df.dropna(subset=need).copy()
df["target_t1"] = df["target_t1"].astype(int)

print(df.shape, "(kỳ vọng 130 cây x 24 tuần = 3120 dòng)")
print(df["target_t1"].value_counts().sort_index())
df.to_csv("data/features.csv", index=False)