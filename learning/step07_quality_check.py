import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print(df.isna().sum().sum())
print(df.dtypes.value_counts())
print(df["Amount"].describe())