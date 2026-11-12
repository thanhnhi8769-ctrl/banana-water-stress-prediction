import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, cohen_kappa_score, mean_absolute_error
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

# ---------- Bộ feature ----------
A = ["NDVI_sg", "NDRE_sg", "NDVI_z", "NDRE_z", "LAI_locf", "altura_locf", "hojas_locf",
     "dNDVI", "dNDRE", "Growth_Rate_Height", "Growth_Rate_Leaves"]
B = ["ET0_week_sum", "ETc", "water_deficit", "VPD_week_mean", "P_week_sum"]
PHYS = ["VPD_per_LAI", "Chlorophyll_Proxy", "Deficit_per_LAI",
        "NDVI_lag1", "NDRE_lag1", "days_since_morfo", "days_since_lai", "weight_stale"]


def no_wd(cols):
    """Bỏ các biến dựa vào water_deficit (kiểm tra rò rỉ treatment)."""
    return [c for c in cols if c not in ("water_deficit", "Deficit_per_LAI")]


FEATURE_SETS = {
    "A_plant": A,
    "B_fao": B,
    "C_plant+fao": A + B,
    "D_full": A + B + PHYS,
    "B_noWD": no_wd(B),
    "C_noWD": no_wd(A + B),
    "D_noWD": no_wd(A + B + PHYS),
}


# ---------- Chia dữ liệu theo cây, phân tầng theo treatment ----------
def load_split(path="data/features.csv", test_frac=0.2, seed=42):
    df = pd.read_csv(path)
    pl = df.groupby("PlantUID")["trat"].first()
    rng = np.random.default_rng(seed)
    test_ids = []
    for _, s in pl.groupby(pl):
        ids = rng.permutation(s.index.to_numpy())
        test_ids += list(ids[: int(round(test_frac * len(ids)))])
    is_test = df["PlantUID"].isin(test_ids)
    return df[~is_test].reset_index(drop=True), df[is_test].reset_index(drop=True)


# ---------- Mô hình ----------
def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000))


def xgb():
    return XGBClassifier(n_estimators=200, max_depth=3, learning_rate=0.1, subsample=0.8,
                         colsample_bytree=0.8, random_state=42, n_jobs=-1)


def _logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p)).reshape(-1, 1)


class FrankHall:
    """Ordinal Frank-Hall: 3 bộ nhị phân P(y>k) ghép thành P(y=k); tùy chọn Platt scaling."""

    def __init__(self, make, calibrate=False):
        self.make, self.calibrate = make, calibrate

    def fit(self, X, y, groups=None):
        self.ms, self.platt = [], []
        for k in range(3):
            yk = (np.asarray(y) > k).astype(int)
            self.ms.append(self.make().fit(X, yk))
            if self.calibrate:
                # Platt fit trên xác suất out-of-fold (group theo cây) của chính tập train
                oof = np.zeros(len(X))
                for a, b in GroupKFold(5).split(X, yk, groups):
                    oof[b] = self.make().fit(X.iloc[a], yk[a]).predict_proba(X.iloc[b])[:, 1]
                self.platt.append(LogisticRegression(C=1e6).fit(_logit(oof), yk))
        return self

    def predict_proba(self, X):
        g = []
        for k, m in enumerate(self.ms):
            p = m.predict_proba(X)[:, 1]
            if self.calibrate:
                p = self.platt[k].predict_proba(_logit(p))[:, 1]
            g.append(p)
        g = np.minimum.accumulate(np.column_stack(g), axis=1)
        return np.column_stack([1 - g[:, 0], g[:, 0] - g[:, 1], g[:, 1] - g[:, 2], g[:, 2]])


MODEL_NAMES = ["MostFrequent", "LogReg", "RandomForest", "XGBoost", "Ordinal_LogReg", "Ordinal_XGB"]


def make_model(name, calibrate=False):
    if name == "MostFrequent":
        return DummyClassifier(strategy="most_frequent")
    if name == "LogReg":
        return lr()
    if name == "RandomForest":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1)
    if name == "XGBoost":
        return xgb()
    if name == "Ordinal_LogReg":
        return FrankHall(lr, calibrate)
    if name == "Ordinal_XGB":
        return FrankHall(xgb, calibrate)
    raise ValueError(name)


def fit_model(m, X, y, groups):
    return m.fit(X, y, groups=groups) if isinstance(m, FrankHall) else m.fit(X, y)


# ---------- Metric ----------
def brier(P, y):
    return float(((P - np.eye(4)[np.asarray(y)]) ** 2).sum(axis=1).mean())


def metrics(y, P):
    p = P.argmax(axis=1)
    return dict(F1=f1_score(y, p, average="macro"),
                QWK=cohen_kappa_score(y, p, weights="quadratic"),
                MAE=mean_absolute_error(y, p),
                Brier=brier(P, y))