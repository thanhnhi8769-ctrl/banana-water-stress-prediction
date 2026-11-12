import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (f1_score, cohen_kappa_score, mean_absolute_error,
                             confusion_matrix)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/features.csv")
A = ["NDVI_mean", "NDRE_mean", "LAI_locf", "altura_locf", "hojas_locf",
     "dNDVI", "dNDRE", "Growth_Rate_Height", "Growth_Rate_Leaves"]
B = ["ET0_week_sum", "ETc", "water_deficit", "VPD_week_mean", "P_week_sum"]
SETS = {"A_plant": A, "B_fao": B, "C_plant+fao": A + B}

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, df["target_t1"], df["PlantUID"]))
train, test = df.iloc[tr_idx], df.iloc[te_idx]
ytr, yte = train["target_t1"], test["target_t1"]


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


print("== KẾT QUẢ TRÊN 26 CÂY TEST ==")
for name, cols in SETS.items():
    m = lr().fit(train[cols], ytr)
    p = m.predict(test[cols])
    print(f"{name:12s} F1={f1_score(yte, p, average='macro'):.3f} "
          f"QWK={cohen_kappa_score(yte, p, weights='quadratic'):.3f} "
          f"MAE={mean_absolute_error(yte, p):.3f}")

cols = A + B
m = lr().fit(train[cols], ytr)
p = m.predict(test[cols])
print("\n== MA TRẬN NHẦM LẪN (hàng = thật, cột = dự đoán) ==")
print(confusion_matrix(yte, p))

print("\n== ĐỘ QUAN TRỌNG CỦA BIẾN (giảm F1 khi xáo trộn) ==")
imp = permutation_importance(m, test[cols], yte, scoring="f1_macro",
                             n_repeats=10, random_state=42)
out = pd.Series(imp.importances_mean, index=cols).sort_values(ascending=False)
print(out.round(4).to_string())