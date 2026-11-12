import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
import os

os.makedirs("outputs/figures", exist_ok=True)
LV = ["Healthy", "Mild", "Moderate", "Severe"]
TR = {1: "T1 (30% ETc)", 2: "T2 (50% ETc)", 3: "T3 (75% ETc)", 4: "T4 (100% ETc)"}
plt.rcParams.update({"figure.dpi": 130, "axes.grid": True, "grid.alpha": 0.3})


def save(name):
    plt.tight_layout()
    plt.savefig(f"outputs/figures/{name}.png")
    plt.close()


# 1. Stress và NDVI theo tuần, theo nhóm tưới
f = pd.read_csv("data/features.csv")
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for t, g in f.groupby("trat"):
    ax[0].plot(*zip(*g.groupby("semana")["estres_nivel"].mean().items()), label=TR[t])
    ax[1].plot(*zip(*g.groupby("semana")["NDVI_sg"].mean().items()), label=TR[t])
ax[0].set(title="Mức stress trung bình theo tuần", xlabel="Tuần", ylabel="0=Healthy ... 3=Severe")
ax[1].set(title="NDVI (đã khử nhiễu) theo tuần", xlabel="Tuần", ylabel="NDVI")
ax[0].legend()
save("fig1_stress_va_ndvi_theo_tuan")

# 2. Ma trận nhầm lẫn (test)
p = pd.read_csv("outputs/test_predictions.csv")
cm = confusion_matrix(p["true_t1"], p["pred_t1"], labels=range(4))
cmn = cm / cm.sum(1, keepdims=True)
fig, ax = plt.subplots(figsize=(5, 4.3))
ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
for i in range(4):
    for j in range(4):
        ax.text(j, i, f"{cm[i, j]}\n({cmn[i, j]:.0%})", ha="center", va="center",
                color="white" if cmn[i, j] > 0.5 else "black", fontsize=9)
ax.set(xticks=range(4), yticks=range(4), xticklabels=LV, yticklabels=LV,
       xlabel="Dự đoán (t+1)", ylabel="Thực tế (t+1)", title="Ma trận nhầm lẫn - tập test")
ax.grid(False)
save("fig2_ma_tran_nham_lan")

# 3. Độ tin cậy xác suất: P(stress >= Moderate)
pr = p["P_Moderate"] + p["P_Severe"]
yt = (p["true_t1"] >= 2).astype(int)
bins = np.linspace(0, 1, 11)
idx = np.clip(np.digitize(pr, bins) - 1, 0, 9)
xs, ys, ns = [], [], []
for b in range(10):
    m = idx == b
    if m.sum() >= 10:
        xs.append(pr[m].mean()); ys.append(yt[m].mean()); ns.append(m.sum())
fig, ax = plt.subplots(figsize=(4.8, 4.5))
ax.plot([0, 1], [0, 1], "k--", label="Hiệu chỉnh hoàn hảo")
ax.scatter(xs, ys, s=np.array(ns) / 4 + 20, c="tab:blue", label="Mô hình")
ax.plot(xs, ys, c="tab:blue", alpha=0.5)
ax.set(xlabel="Xác suất dự đoán P(Moderate hoặc Severe)", ylabel="Tần suất thực tế",
       title="Độ tin cậy xác suất - tập test", xlim=(0, 1), ylim=(0, 1))
ax.legend()
save("fig3_calibration")

# 4. Quyết định tưới theo tỷ lệ chi phí
d = pd.read_csv("outputs/decision.csv")
pols = list(dict.fromkeys(d["policy"]))
Rs = list(dict.fromkeys(d["R"]))
w = 0.8 / len(pols)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for k, pol in enumerate(pols):
    s = d[d["policy"] == pol].set_index("R").loc[Rs]
    ax[0].bar(np.arange(len(Rs)) + k * w, s["cost"], w, label=pol)
    ax[1].bar(np.arange(len(Rs)) + k * w, s["under"] * 100, w, label=pol)
for a in ax:
    a.set_xticks(np.arange(len(Rs)) + 0.4 - w / 2)
    a.set_xticklabels(Rs)
    a.set_xlabel("Tỷ lệ chi phí thiếu nước : tưới thừa")
ax[0].set(yscale="log", ylabel="Chi phí TB mỗi cây-tuần (log)", title="Chi phí theo chiến lược")
ax[1].set(ylabel="% cây-tuần bị thiếu nước", title="Tỷ lệ thiếu nước")
ax[0].legend(fontsize=8)
save("fig4_quyet_dinh_tuoi")

# 5. Độ quan trọng của biến
fi = pd.read_csv("outputs/feature_importance.csv", index_col=0).iloc[:, 0].sort_values().tail(12)
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.barh(fi.index, fi.values, color="tab:green")
ax.set(xlabel="Giảm F1 khi xáo trộn biến", title="Độ quan trọng của biến (test)")
save("fig5_do_quan_trong_bien")

# 6. Ablation (QWK theo bộ feature và model)
ab = pd.read_csv("outputs/ablation_folds.csv")
ab = ab[ab["model"] != "MostFrequent"]
st = ab.groupby(["set", "model"])["QWK"].agg(["mean", "std"]).reset_index()
sets = list(dict.fromkeys(ab["set"]))
models = list(dict.fromkeys(ab["model"]))
w = 0.8 / len(models)
fig, ax = plt.subplots(2, 1, figsize=(10, 7))
for k, m in enumerate(models):
    s = st[st["model"] == m].set_index("set").loc[sets]
    for a in ax:
        a.bar(np.arange(len(sets)) + k * w, s["mean"], w, yerr=s["std"], capsize=2, label=m)
for a in ax:
    a.set_xticks(np.arange(len(sets)) + 0.4 - w / 2)
    a.set_xticklabels(sets)
    a.set_ylabel("QWK (CV 5 fold)")
ax[0].set(title="Ablation: QWK theo bộ feature", ylim=(0, 1))
ax[0].legend(fontsize=8, ncol=3)
z = st[st["set"] != "B_noWD"]
ax[1].set(ylim=(z["mean"].min() - 0.03, min(1, z["mean"].max() + 0.03)), title="Phóng to (bỏ B_noWD)")
ax[1].set_xlim(-0.2, len(sets) - 0.2)
save("fig6_ablation")

print("Đã lưu 6 hình vào outputs/figures/")
