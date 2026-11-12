"""Kiểm tra dữ liệu thật trước khi chạy pipeline: nhãn, rò rỉ, water_deficit, cấu trúc."""
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from config import RAW_PATH, ENCODING, RENAME, LABEL_COL, LABEL_ORDER, REQUIRED

out = []


def log(s=""):
    print(s)
    out.append(str(s))


def warn(s):
    log("  [CẢNH BÁO] " + s)


def cramers_v(a, b):
    t = pd.crosstab(a, b).to_numpy(float)
    e = t.sum(1, keepdims=True) * t.sum(0, keepdims=True) / t.sum()
    chi2 = ((t - e) ** 2 / np.where(e == 0, 1, e)).sum()
    return float(np.sqrt(chi2 / (t.sum() * (min(t.shape) - 1))))


df = pd.read_csv(RAW_PATH, encoding=ENCODING).rename(columns=RENAME)

log("== 1. CẤU TRÚC ==")
log(f"Kích thước: {df.shape}")
miss = [c for c in REQUIRED + [LABEL_COL] if c not in df.columns]
if miss:
    warn(f"Thiếu cột {miss} -> sửa RENAME trong config.py rồi chạy lại")
    raise SystemExit(1)
log(f"Số cây: {df['PlantUID'].nunique()} | số tuần: {df['semana'].nunique()} (kỳ vọng 130 và 26)")
rows = df.groupby("PlantUID").size()
log(f"Số dòng mỗi cây: min={rows.min()}, max={rows.max()}")
if rows.nunique() > 1:
    warn("Số tuần mỗi cây không đều")
if df.duplicated(["PlantUID", "semana"]).any():
    warn("Có dòng trùng (PlantUID, semana)")
log("Tỷ lệ thiếu cao nhất:\n" + df[REQUIRED].isna().mean().sort_values(ascending=False).head(5).round(3).to_string())

log("\n== 2. THỜI TIẾT (kỳ vọng giống nhau cho mọi cây trong tuần) ==")
for c in ["ET0_week_sum", "VPD_week_mean", "P_week_sum", "ETc"]:
    s = df.groupby("semana")[c].std().max()
    log(f"{c}: std lớn nhất giữa các cây cùng tuần = {s:.4f}")
    if s > 1e-6:
        warn(f"{c} khác nhau giữa các cây cùng tuần")

# ---- nhãn ----
y = df[LABEL_COL]
if pd.api.types.is_numeric_dtype(y):
    lab = y.astype(int)
else:
    lab = y.astype(str).str.strip().str.lower().map({s.lower(): i for i, s in enumerate(LABEL_ORDER)})
df["_y"] = lab

log(f"\n== 3. NHÃN '{LABEL_COL}' ==")
log("Phân bố:\n" + df["_y"].value_counts().sort_index().to_string())
log("Bảng chéo trat x nhãn:\n" + pd.crosstab(df["trat"], df["_y"]).to_string())
v = cramers_v(df["trat"], df["_y"])
log(f"Cramér's V giữa trat và nhãn = {v:.3f}")
acc_t = (df.groupby("trat")["_y"].transform(lambda s: s.mode().iloc[0]) == df["_y"]).mean()
acc_ts = (df.groupby(["trat", "semana"])["_y"].transform(lambda s: s.mode().iloc[0]) == df["_y"]).mean()
log(f"Đoán nhãn chỉ bằng trat: đúng {acc_t:.1%} | bằng (trat, tuần): đúng {acc_ts:.1%}")
if acc_ts > 0.9:
    warn("Nhãn gần như do (treatment, tuần) quyết định -> nhiều khả năng là treatment đổi tên, không phải stress độc lập")
d = df.sort_values(["PlantUID", "semana"]).copy()
nxt = d.groupby("PlantUID")["_y"].shift(-1)
ok = nxt.notna()
log(f"Persistence (đoán t+1 = nhãn hiện tại): đúng {(nxt[ok] == d['_y'][ok]).mean():.1%}  <- mốc mà dự báo sớm phải vượt")

log("\n== 4. NHÃN CÓ SUY RA TỪ MỘT CỘT KHÔNG? (cây quyết định độ sâu 3, CV theo cây, độ chính xác) ==")
skip = {"PlantUID", "semana", "_y", LABEL_COL}
cand = [c for c in df.select_dtypes("number").columns if c not in skip and not c.lower().startswith("estres")]
sc = {}
gkf = GroupKFold(5)
for c in cand:
    X = df[[c]].fillna(df[c].median())
    sc[c] = cross_val_score(DecisionTreeClassifier(max_depth=3, random_state=0), X, df["_y"],
                            groups=df["PlantUID"], cv=gkf).mean()
sc = pd.Series(sc).sort_values(ascending=False)
log(sc.head(8).round(3).to_string())
for c, a in sc.items():
    if a > 0.95:
        warn(f"Cột '{c}' một mình đoán đúng {a:.1%} nhãn -> nhãn có thể được tạo từ chính cột này (circular leakage)")

log("\n== 5. CỘT QUÁ GIỐNG TREATMENT (|r| với trat > 0.9, cần loại khỏi feature) ==")
oth = [c for c in cand if c != "trat"]
cr = df[oth + ["trat"]].corr()["trat"].drop("trat").abs().sort_values(ascending=False)
log(cr.head(10).round(3).to_string())
for c, r in cr.items():
    if r > 0.9:
        warn(f"'{c}' |r|={r:.3f} với trat" + (" (đang nằm trong REQUIRED!)" if c in REQUIRED else ""))

log("\n== 6. water_deficit ==")
wd = df["water_deficit"]
eta = []
for _, g in df.groupby("semana"):
    tot = ((g["water_deficit"] - g["water_deficit"].mean()) ** 2).sum()
    if tot > 0:
        bet = g.groupby("trat")["water_deficit"].agg(["count", "mean"])
        eta.append((bet["count"] * (bet["mean"] - g["water_deficit"].mean()) ** 2).sum() / tot)
e = float(np.mean(eta)) if eta else 0.0
log(f"Tỷ lệ phương sai do treatment (eta^2, TB theo tuần) = {e:.3f}")
if e > 0.5:
    warn("water_deficit phụ thuộc mạnh vào treatment -> dùng bộ _noWD (FORCE trong 11_final.py) hoặc tính lại từ thời tiết + Kc")
inc = df.sort_values(["PlantUID", "semana"]).groupby("PlantUID")["water_deficit"].diff()
log(f"Tỷ lệ bước giảm của water_deficit theo thời gian: {(inc < 0).mean():.1%} (nếu cộng dồn thì ~0%)")
if "riego_mm" in df.columns:
    log(f"corr(Δwater_deficit, riego_mm) = {inc.corr(df['riego_mm']):.3f}")

log("\n== 7. DỮ LIỆU LẶP (LOCF) ==")
for c in ("days_since_morfo", "days_since_lai"):
    log(f"{c} > 0: {(df[c] > 0).mean():.1%} số dòng")

with open("outputs/validation_report.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(out))
print("\nĐã lưu outputs/validation_report.txt")
