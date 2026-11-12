import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

df = pd.read_csv("data/features.csv")

A = ["NDVI_mean", "NDRE_mean", "LAI_locf", "altura_locf", "hojas_locf",
     "dNDVI", "dNDRE", "Growth_Rate_Height", "Growth_Rate_Leaves"]
B = ["ET0_week_sum", "ETc", "water_deficit", "VPD_week_mean", "P_week_sum"]
PHYS = ["VPD_per_LAI", "Chlorophyll_Proxy", "Deficit_per_LAI",
        "days_since_morfo", "days_since_lai"]
SETS = {"A_plant": A, "C_plant+fao": A + B, "D_full": A + B + PHYS}

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, _ = next(gss.split(df, df["target_t1"], df["PlantUID"]))
train = df.iloc[tr_idx].reset_index(drop=True)


class FrankHall:
    """Ordinal: K-1 bộ phân loại nhị phân P(y > k), ghép lại thành P(y = k)."""
    def __init__(self, make):
        self.make = make

    def fit(self, X, y):
        self.ms = [self.make().fit(X, (y > k).astype(int)) for k in range(3)]
        return self

    def predict_proba(self, X):
        g = np.column_stack([m.predict_proba(X)[:, 1] for m in self.ms])
        g = np.minimum.accumulate(g, axis=1)  # đảm bảo P(y>k) giảm dần
        p = np.column_stack([1 - g[:, 0], g[:, 0] - g[:, 1],
                             g[:, 1] - g[:, 2], g[:, 2]])
        return np.clip(p, 0, 1)

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)


models = {
    "Ordinal_LogReg": lambda: FrankHall(
        lambda: make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))),
    "Ordinal_XGB": lambda: FrankHall(
        lambda: XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1,
                              random_state=42, eval_metric="logloss")),
}


def score(y, p):
    return (f1_score(y, p, average="macro"),
            cohen_kappa_score(y, p, weights="quadratic"),
            mean_absolute_error(y, p))


cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
rows = []
for sname, cols in SETS.items():
    for mname, make in models.items():
        res = []
        for a, b in cv.split(train, train["target_t1"], train["PlantUID"]):
            m = make().fit(train.loc[a, cols], train.loc[a, "target_t1"].values)
            res.append(score(train.loc[b, "target_t1"], m.predict(train.loc[b, cols])))
        r = np.array(res)
        rows.append(dict(features=sname, model=mname,
                         F1=f"{r[:,0].mean():.3f}±{r[:,0].std():.3f}",
                         QWK=f"{r[:,1].mean():.3f}±{r[:,1].std():.3f}",
                         MAE=f"{r[:,2].mean():.3f}±{r[:,2].std():.3f}"))

print(pd.DataFrame(rows).to_string(index=False))