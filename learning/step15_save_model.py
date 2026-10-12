import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, precision_score
from xgboost import XGBClassifier

# load the data and make the same split as before (run from the fraudshield folder)
df = pd.read_csv("data/creditcard.csv")
X = df.drop(columns=["Class"])
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# train the final model on ALL of the training part
model = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                      random_state=42, eval_metric="logloss")
model.fit(X_train, y_train)

# save it to a file
os.makedirs("models", exist_ok=True)
model.save_model("models/xgb_fraud.json")
print("saved: models/xgb_fraud.json,", round(os.path.getsize("models/xgb_fraud.json") / 1024), "KB")

# load it back into a brand new model object, no training
loaded = XGBClassifier()
loaded.load_model("models/xgb_fraud.json")

# check: the loaded model must give exactly the same scores
same = np.allclose(model.predict_proba(X_test)[:, 1], loaded.predict_proba(X_test)[:, 1])
print("loaded model gives the same scores:", same)

# final exam, once, at cutoff 0.5
pred = (loaded.predict_proba(X_test)[:, 1] >= 0.5).astype(int)
print("test recall:", round(recall_score(y_test, pred), 3),
      "precision:", round(precision_score(y_test, pred), 3))