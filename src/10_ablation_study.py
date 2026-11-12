import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from common import FEATURE_SETS, MODEL_NAMES, load_split, make_model, fit_model, metrics

train, _ = load_split()
y, groups = train["target_t1"].values, train["PlantUID"].values
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(train, y, groups))
print("Cây train:", train["PlantUID"].nunique())

rows = []
for sname, cols in FEATURE_SETS.items():
    for mname in MODEL_NAMES:
        for k, (a, b) in enumerate(folds):
            m = fit_model(make_model(mname), train.loc[a, cols], y[a], groups[a])
            P = m.predict_proba(train.loc[b, cols])
            rows.append(dict(set=sname, model=mname, fold=k, **metrics(y[b], P)))
    print("xong", sname, flush=True)

res = pd.DataFrame(rows)
res.to_csv("outputs/ablation_folds.csv", index=False)

M = ["F1", "QWK", "MAE", "Brier"]
s = res.groupby(["set", "model"], sort=False)[M].agg(["mean", "std"])
out = pd.DataFrame({m: s[m]["mean"].map("{:.3f}".format) + "±" + s[m]["std"].map("{:.3f}".format) for m in M})
out.to_csv("outputs/ablation_summary.csv")
print("\n== BẢNG ABLATION (CV 5 fold theo cây) ==")
print(out.to_string())

q = res.pivot_table(index=["model", "fold"], columns="set", values="QWK")
d = pd.DataFrame({"C-A (FAO thêm vào cây)": q["C_plant+fao"] - q["A_plant"],
                  "D-C (physics thêm)": q["D_full"] - q["C_plant+fao"],
                  "C-C_noWD (đóng góp water_deficit)": q["C_plant+fao"] - q["C_noWD"],
                  "D_noWD-A (FAO không WD)": q["D_noWD"] - q["A_plant"]})
print("\n== CHÊNH LỆCH QWK THEO FOLD (mean, std) ==")
print(d.groupby("model").agg(["mean", "std"]).round(3).T.to_string())