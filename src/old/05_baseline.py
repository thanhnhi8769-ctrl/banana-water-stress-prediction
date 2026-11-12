import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/features.csv")

A = ["NDVI_mean", "NDRE_mean", "LAI_locf", "altura_locf", "hojas_locf",
     "dNDVI", "dNDRE", "Growth_Rate_Height", "Growth_Rate_Leaves"]
B = ["ET0_week_sum", "ETc", "water_deficit", "VPD_week_mean", "P_week_sum"]
SETS = {"A_plant": A, "B_fao": B, "C_plant+fao": A + B}

# Test: 20% số cây
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, te_idx = next(gss.split(df, df["target_t1"], df["PlantUID"]))
train, test = df.iloc[tr_idx].reset_index(drop=True), df.iloc[te_idx]
print("Cây train:", train["PlantUID"].nunique(), "| cây test:", test["PlantUID"].nunique())

def score(y, p):
    return (f1_score(y, p, average="macro"),
            cohen_kappa_score(y, p, weights="quadratic"),
            mean_absolute_error(y, p))

models = {
    "MostFrequent": lambda: DummyClassifier(strategy="most_frequent"),
    "LogReg": lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)),
}

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for sname, cols in SETS.items():
    for mname, make in models.items():
        res = []
        for a, b in cv.split(train, train["target_t1"], train["PlantUID"]):
            m = make().fit(train.loc[a, cols], train.loc[a, "target_t1"])
            res.append(score(train.loc[b, "target_t1"], m.predict(train.loc[b, cols])))
        r = np.array(res)
        rows.append(dict(features=sname, model=mname,
                         F1=f"{r[:,0].mean():.3f}±{r[:,0].std():.3f}",
                         QWK=f"{r[:,1].mean():.3f}±{r[:,1].std():.3f}",
                         MAE=f"{r[:,2].mean():.3f}±{r[:,2].std():.3f}"))

print(pd.DataFrame(rows).to_string(index=False))