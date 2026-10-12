import numpy as np
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

# load the data from your Mac (run this from the fraudshield folder)
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]

# same split as before
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# train XGBoost and get its fraud scores on the test set
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                    random_state=42, eval_metric="logloss")
xgb.fit(X_train, y_train)
proba_x = xgb.predict_proba(X_test)[:, 1]

# SHAP: why did the model score one transaction the way it did?
explainer = shap.TreeExplainer(xgb)

# first real fraud in the test set that the model flagged at cutoff 0.5
idx = np.where((proba_x >= 0.5) & (y_test.values == 1))[0][0]
row = X_test.iloc[[idx]]

sv = explainer.shap_values(row)[0]
contrib = pd.Series(sv, index=X_test.columns).sort_values(key=abs, ascending=False)
print("Top 5 columns for this transaction (+ pushes toward fraud, - away):")
print(contrib.head(5).round(3))

# check: starting point + all contributions must give back the model's own score
total = float(explainer.expected_value) + sv.sum()
print("start + contributions, turned into a score:", round(1 / (1 + np.exp(-total)), 4))
print("model's own score:                          ", round(float(proba_x[idx]), 4))

# ---- global view: which columns matter most across many transactions ----
# all frauds in the test set + 500 genuine ones, so fraud is well represented
fraud_rows = X_test[y_test == 1]
genuine_rows = X_test[y_test == 0].sample(500, random_state=42)
sample = pd.concat([fraud_rows, genuine_rows])

sv_all = explainer.shap_values(sample)    # one row of contributions per transaction
overall = pd.Series(np.abs(sv_all).mean(axis=0), index=X_test.columns)
print("Most important columns overall (average size of push):")
print(overall.sort_values(ascending=False).head(5).round(3))