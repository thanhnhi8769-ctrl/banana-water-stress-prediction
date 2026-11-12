import numpy as np
import pandas as pd

LAMBDA = 0.1  # tốc độ giảm trọng số dữ liệu cũ (theo ngày)

df = pd.read_csv("data/clean.csv").sort_values(["PlantUID", "semana"]).reset_index(drop=True)


def causal_sg(s, window=5, order=2):
    """Savitzky-Golay nhân quả: giá trị tuần t chỉ dùng các tuần <= t (không rò rỉ tương lai)."""
    v = s.to_numpy(float)
    out = v.copy()
    for t in range(len(v)):
        w = v[max(0, t - window + 1): t + 1]
        if len(w) >= 3:
            x = np.arange(len(w))
            out[t] = np.polyval(np.polyfit(x, w, min(order, len(w) - 2)), x[-1])
    return pd.Series(out, index=s.index)


for c, name in [("NDVI_mean", "NDVI"), ("NDRE_mean", "NDRE")]:
    df[f"{name}_sg"] = df.groupby("PlantUID")[c].transform(causal_sg)
    # Z-score theo tuần (đợt bay): chỉ dùng dữ liệu cùng tuần, không dùng nhãn
    g = df.groupby("semana")[f"{name}_sg"]
    df[f"{name}_z"] = (df[f"{name}_sg"] - g.transform("mean")) / g.transform("std")

df["weight_stale"] = np.exp(-LAMBDA * df["days_since_lai"])

print(df[["NDVI_mean", "NDVI_sg", "NDVI_z", "NDRE_sg", "weight_stale"]].describe().round(3))
print("Độ nhiễu NDVI (std của diff) trước/sau SG:",
      round(df.groupby("PlantUID")["NDVI_mean"].diff().std(), 4),
      round(df.groupby("PlantUID")["NDVI_sg"].diff().std(), 4))
df.to_csv("data/denoised.csv", index=False)