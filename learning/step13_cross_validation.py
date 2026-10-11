import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# load the data from your Mac (run this from the fraudshield folder)
df = pd.read_csv("data/creditcard.csv")

X = df.drop(columns=["Class"])
y = df["Class"]

# same split as before: the test set stays hidden and is NOT used here
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# 5 stratified folds inside the training part only
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = [
    ("XGBoost", XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                              random_state=42, eval_metric="logloss")),
    ("RandomForest", RandomForestClassifier(n_estimators=200, n_jobs=-1,
                                            random_state=42)),
]

for name, m in models:
    scores = cross_val_score(m, X_train, y_train, cv=cv, scoring="average_precision")
    print(name, [round(float(s), 3) for s in scores],
          "mean:", round(float(scores.mean()), 3), "std:", round(float(scores.std()), 3))