import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from common import FEATURE_SETS, load_split, make_model, fit_model, metrics, brier

FORCE = None  # ví dụ: ("D_noWD", "Ordinal_LogReg"); None = tự chọn theo QWK CV

train, test = load_split()
y, groups = train["target_t1"].values, train["PlantUID"].values
yte = test["target_t1"].values

# ---------- 1. Chọn cấu hình tốt nhất (model ordinal, theo QWK trung bình CV) ----------
if FORCE:
    sname, mname = FORCE
else:
    ab = pd.read_csv("outputs/ablation_folds.csv")
    ab = ab[ab["model"].str.startswith("Ordinal")]
    sname, mname = ab.groupby(["set", "model"])["QWK"].mean().idxmax()
cols = FEATURE_SETS[sname]
print(f"Cấu hình chọn: bộ feature = {sname} | model = {mname} | {len(cols)} biến")

# ---------- 2. Calibration: so sánh trước/sau, Platt fit bên trong từng fold ----------
folds = list(StratifiedGroupKFold(5, shuffle=True, random_state=42).split(train, y, groups))
oof = {False: np.zeros((len(train), 4)), True: np.zeros((len(train), 4))}
for cal in (False, True):
    for a, b in folds:
        m = fit_model(make_model(mname, calibrate=cal), train.loc[a, cols], y[a], groups[a])
        oof[cal][b] = m.predict_proba(train.loc[b, cols])
b_raw, b_cal = brier(oof[False], y), brier(oof[True], y)
use_cal = b_cal < b_raw
print(f"\nBrier CV: raw={b_raw:.4f} | calibrated={b_cal:.4f} -> dùng {'calibrated' if use_cal else 'raw'}")

# ---------- 3. Huấn luyện toàn bộ train, đánh giá MỘT LẦN trên test ----------
model = fit_model(make_model(mname, calibrate=use_cal), train[cols], y, groups)
Pte = model.predict_proba(test[cols])
pred = Pte.argmax(axis=1)
r = metrics(yte, Pte)
print("\n== KẾT QUẢ TRÊN TẬP TEST (%d cây) ==" % test["PlantUID"].nunique())
print({k: round(v, 3) for k, v in r.items()})
pers = test["estres_nivel"].values  # mốc đối chứng: đoán t+1 = mức stress hiện tại (model không được thấy cột này)
print("Persistence (t+1 = hiện tại): F1=%.3f | QWK=%.3f | MAE=%.3f" % (
    f1_score(yte, pers, average="macro"), cohen_kappa_score(yte, pers, weights="quadratic"),
    np.abs(yte - pers).mean()))

rng = np.random.default_rng(0)
plants = test["PlantUID"].unique()
idx = {p: np.where(test["PlantUID"].values == p)[0] for p in plants}
qs, fs = [], []
for _ in range(500):
    ii = np.concatenate([idx[p] for p in rng.choice(plants, len(plants))])
    qs.append(cohen_kappa_score(yte[ii], pred[ii], weights="quadratic"))
    fs.append(f1_score(yte[ii], pred[ii], average="macro"))
print("KTC 95%% (bootstrap theo cây): QWK [%.3f, %.3f] | F1 [%.3f, %.3f]"
      % (*np.percentile(qs, [2.5, 97.5]), *np.percentile(fs, [2.5, 97.5])))
print("\nMa trận nhầm lẫn (hàng = thật, cột = dự đoán):")
print(confusion_matrix(yte, pred))

# ---------- 4. Độ quan trọng biến (permutation, trên test) ----------
base = f1_score(yte, pred, average="macro")
imp = {}
for c in cols:
    d = []
    for _ in range(10):
        Xp = test[cols].copy()
        Xp[c] = rng.permutation(Xp[c].values)
        d.append(base - f1_score(yte, model.predict_proba(Xp).argmax(axis=1), average="macro"))
    imp[c] = np.mean(d)
imp = pd.Series(imp).sort_values(ascending=False)
print("\n== ĐỘ QUAN TRỌNG BIẾN (giảm F1) ==")
print(imp.round(4).head(10).to_string())
imp.to_csv("outputs/feature_importance.csv")

# ---------- 5. Tầng 2: quyết định tưới theo expected cost ----------
ACTIONS = np.array([0.0, 0.5, 1.0])         # 0%, 50%, 100% ETc
NEED = np.array([0.0, 0.25, 0.5, 1.0])      # nhu cầu nước theo mức stress (Healthy..Severe)


def cost_matrix(R):
    """C[a, c]: tưới thừa tốn 1/đơn vị ETc; thiếu nước tốn R/đơn vị ETc."""
    a, n = ACTIONS[:, None], NEED[None, :]
    return np.maximum(a - n, 0) + R * np.maximum(n - a, 0)


ETC, RAIN = test["ETc"].values, test["P_week_sum"].values  # tuần t làm xấp xỉ cho t+1
rows = []
for R in [5, 10, 20, 50]:
    C = cost_matrix(R)
    policies = {
        "Expected-cost": (Pte @ C.T).argmin(axis=1),
        "Argmax lớp":    np.array([0, 1, 1, 2])[pred],
        "Luôn 100% ETc": np.full(len(test), 2),
        "Không tưới":    np.zeros(len(test), int),
    }
    for name, a in policies.items():
        mm = np.maximum(ACTIONS[a] * ETC - RAIN, 0)
        rows.append(dict(R=f"{R}:1", policy=name,
                         cost=C[a, yte].mean(),
                         under=(ACTIONS[a] < NEED[yte]).mean(),
                         over=(ACTIONS[a] > NEED[yte]).mean(),
                         water_pct_ETc=ACTIONS[a].mean() * 100,
                         mm_week=mm.mean()))
dec = pd.DataFrame(rows)
dec.to_csv("outputs/decision.csv", index=False)
print("\n== QUYẾT ĐỊNH TƯỚI TRÊN TEST (cost = chi phí TB mỗi cây-tuần) ==")
print(dec.round(3).to_string(index=False))

out = test[["PlantUID", "semana"]].copy()
out["true_t1"] = yte
out["pred_t1"] = pred
out[[f"P_{k}" for k in ("Healthy", "Mild", "Moderate", "Severe")]] = Pte.round(4)
out.to_csv("outputs/test_predictions.csv", index=False)
