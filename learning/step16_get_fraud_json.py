import json
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"]); y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

fraud_rows = X_test[y_test == 1]
row = fraud_rows.iloc[0]
print(json.dumps(row.to_dict(), indent=2))