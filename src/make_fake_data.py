import os

import numpy as np
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

rng = np.random.default_rng(42)
N_PLANTS, N_WEEKS = 130, 26

# Thời tiết theo tuần (giống nhau cho mọi cây)
weeks = np.arange(1, N_WEEKS + 1)
et0 = rng.normal(32, 4, N_WEEKS).clip(20)
vpd = rng.normal(1.1, 0.2, N_WEEKS).clip(0.4)
rain = rng.gamma(1.2, 12, N_WEEKS) * (rng.random(N_WEEKS) < 0.5)
kc = np.interp(weeks, [1, 8, 18, 26], [0.5, 0.9, 1.1, 1.0])
etc_w = et0 * kc

frac_map = {1: 0.30, 2: 0.50, 3: 0.75, 4: 1.00}
rows = []
for p in range(N_PLANTS):
    trat = p % 4 + 1
    frac = frac_map[trat]
    plant_eff = rng.normal(0, 0.05)
    cum_def = 0.0
    held = None
    for i, w in enumerate(weeks):
        riego = frac * etc_w[i]
        deficit = max(etc_w[i] - riego - rain[i], 0)
        cum_def += deficit
        stress = (1 - frac) * (1 - np.exp(-w / 8)) + plant_eff + rng.normal(0, 0.04)
        level = int(np.digitize(stress, [0.15, 0.35, 0.55]))  # 0=Healthy..3=Severe

        ndvi = 0.85 - 0.35 * stress + rng.normal(0, 0.02)
        ndre = 0.50 - 0.25 * stress + rng.normal(0, 0.015)
        lai = max(0.3, 0.2 + 0.12 * w * (1 - 0.6 * stress) + rng.normal(0, 0.1))
        altura = 30 + 9 * w * (1 - 0.5 * stress) + rng.normal(0, 4)
        hojas = 4 + 0.35 * w * (1 - 0.4 * stress) + rng.normal(0, 0.4)

        if w % 2 == 1 or held is None:      # đo định kỳ 2 tuần/lần, tuần lẻ có đo
            held = (lai, altura, hojas)
        days_since = 0 if w % 2 == 1 else 7

        rows.append(dict(
            PlantUID=f"P{p+1:03d}", semana=w, trat=trat,
            fila=p // 13 + 1, columna=p % 13 + 1,
            NDVI_mean=ndvi, NDRE_mean=ndre,
            LAI_locf=held[0], altura_locf=held[1], hojas_locf=held[2],
            days_since_morfo=days_since, days_since_lai=days_since,
            ET0_week_sum=et0[i], VPD_week_mean=vpd[i], P_week_sum=rain[i],
            ETc=etc_w[i], water_deficit=cum_def,
            soil_moisture_0_100=20 + 6 * trat + rng.normal(0, 0.8),
            VWC_pct=18 + 5 * trat + rng.normal(0, 0.8),
            riego_mm=riego, f_depl=1 - frac, s_norm=stress,
            estres_nivel=level, estresG_nivel=level,
        ))

df = pd.DataFrame(rows)
df.loc[rng.random(len(df)) < 0.03, "NDVI_mean"] = np.nan  # thêm vài ô thiếu
df.to_csv("data/raw/dataset_operacional_final.csv", index=False)
print(df.shape)