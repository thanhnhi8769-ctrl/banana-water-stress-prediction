import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("data/features.csv")
A = ["NDVI_mean", "NDRE_mean", "LAI_locf", "altura_locf", "hojas_locf",
     "dNDVI", "dNDRE", "Growth_Rate_Height", "Growth_Rate_Leaves"]
B = ["ET0_week_sum", "ETc", "water_deficit", "VPD_week_mean", "P_week_sum"]
cols = A + B

gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, _ = next(gss.split(df, df["target_t1"], df["PlantUID"]))
train = df.iloc[tr_idx].reset_index(drop=True)
y = train["target_t1"].values


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


def frank_hall(g):
    g = np.minimum.accumulate(g, axis=1)
    p = np.column_stack([1 - g[:, 0], g[:, 0] - g[:, 1], g[:, 1] - g[:, 2], g[:, 2]])
    return np.clip(p, 0, 1)


def fit_predict(Xa, ya, Xb, calibrate):
    """Trả về P(y>k) cho Xb. Nếu calibrate: Platt scaling fit trên tập con riêng."""
    out = []
    for k in range(3):
        yk = (ya > k).astype(int)
        if calibrate:
            # chia nhỏ tập train: 75% fit model, 25% fit Platt (không dùng fold kiểm tra)
            n = len(Xa)
            idx = np.random.default_rng(k).permutation(n)
            fit_i, cal_i = idx[: int(0.75 * n)], idx[int(0.75 * n):]
            m = lr().fit(Xa.iloc[fit_i], yk[fit_i])
            s = m.decision_function(Xa.iloc[cal_i]).reshape(-1, 1)
            platt = LogisticRegression().fit(s, yk[cal_i])
            out.append(platt.predict_proba(m.decision_function(Xb).reshape(-1, 1))[:, 1])
        else:
            m = lr().fit(Xa, yk)
            out.append(m.predict_proba(Xb)[:, 1])
    return np.column_stack(out)


def cv_proba(calibrate):
    P = np.zeros((len(train), 4))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for a, b in cv.split(train, y, train["PlantUID"]):
        g = fit_predict(train.loc[a, cols], y[a], train.loc[b, cols], calibrate)
        P[b] = frank_hall(g)
    P /= P.sum(axis=1, keepdims=True)
    return P


def brier(P, y):
    Y = np.eye(4)[y]
    return ((P - Y) ** 2).sum(axis=1).mean()


P_raw, P_cal = cv_proba(False), cv_proba(True)
print("Brier trước calibration:", round(brier(P_raw, y), 4))
print("Brier sau  calibration:", round(brier(P_cal, y), 4))

# --- Tầng 2: quyết định tưới theo expected cost ---
# Hành động: 0 = không tưới, 1 = tưới.
# Chi phí mỗi lớp stress (Healthy, Mild, Moderate, Severe):
#  - không tưới: bỏ sót stress -> chi phí tăng theo mức, nhân với tỷ lệ R
#  - tưới: tốn nước, chi phí 1 khi Healthy/Mild, thấp (0) khi Moderate/Severe
true_need = (y >= 2).astype(int)  # cần tưới thật sự nếu Moderate/Severe
print("\nTỷ lệ chi phí | Tưới | Bỏ sót | Tưới thừa | Tổng chi phí")
for R in [5, 10, 20, 50]:
    cost_no = np.array([0, 0, 1.0 * R, 2.0 * R])  # không tưới: bỏ sót Moderate/Severe
    cost_yes = np.array([1.0, 1.0, 0.0, 0.0])     # tưới: tốn nước khi Healthy/Mild
    ec_no = P_cal @ cost_no
    ec_yes = P_cal @ cost_yes
    act = (ec_yes < ec_no).astype(int)
    miss = int(((act == 0) & (true_need == 1)).sum())
    waste = int(((act == 1) & (true_need == 0)).sum())
    total = np.where(act == 1, cost_yes[y], cost_no[y]).sum()
    print(f"{R:>5}:1       | {act.mean():.1%} | {miss:>6} | {waste:>9} | {total:,.0f}")