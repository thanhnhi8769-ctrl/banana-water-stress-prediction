import pandas as pd

pd.set_option("display.width", 200)
df = pd.read_csv("data/raw/dataset_operacional_final.csv")

print("== KÍCH THƯỚC ==", df.shape)
print("== SỐ CÂY ==", df["PlantUID"].nunique())
print("== SỐ TUẦN ==", df["semana"].nunique())

print("\n== TẤT CẢ CÁC CỘT ==")
print(df.dtypes.to_string())

print("\n== TỶ LỆ THIẾU (top 20) ==")
print(df.isna().mean().sort_values(ascending=False).head(20))

print("\n== CÁC CỘT estres ==")
est = [c for c in df.columns if c.lower().startswith("estres")]
for c in est:
    print(c, "->", df[c].nunique(), "giá trị:", df[c].value_counts().head(6).to_dict())

print("\n== estres vs trat ==")
for c in est:
    print(c)
    print(pd.crosstab(df["trat"], df[c]))

print("\n== TƯƠNG QUAN CỘT SỐ VỚI trat ==")
num = df.select_dtypes("number")
print(num.corr()["trat"].abs().sort_values(ascending=False).head(15))