import pandas as pd

df = pd.DataFrame({
    "amount":   [500, 1200, 300, 800, 2000, 150, 700, 60, 9000, 15000, 8000, 12000],
    "hour":     [14, 11, 16, 10, 15, 18, 12, 20, 3, 2, 4, 1],
    "is_fraud": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
})


print(df)
print(df.shape)
print(df["is_fraud"].value_counts())

fraud_rows = df[df["is_fraud"] == 1]
print(fraud_rows)
print("Average amount, fraud:", fraud_rows["amount"].mean())
print("Average amount, genuine:", df[df["is_fraud"] == 0]["amount"].mean())

df.to_csv("data/tiny.csv", index=False)
df2 = pd.read_csv("data/tiny.csv")
print(df2.shape)
print(df2.equals(df))