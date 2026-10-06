import pandas as pd

df = pd.read_csv("data/creditcard.csv")

print(df.shape)
print(df.head())
print(df["Class"].value_counts())
print(df["Class"].mean())


