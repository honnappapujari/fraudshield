import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import (accuracy_score, recall_score, precision_score,
                             confusion_matrix, average_precision_score)
from xgboost import XGBClassifier

# ---- Step 1: load the data ----
df = pd.read_csv("data/creditcard.csv")
print(df.shape)
print(df["Class"].value_counts())

# ---- Step 2: features vs label, then a stratified split ----
X = df.drop(columns=["Class"])   # never put the label in X (data leakage)
y = df["Class"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print("train frauds:", y_train.sum(), "test frauds:", y_test.sum())


def report(name, y_true, pred):
    """Print accuracy, recall, precision and the confusion matrix."""
    print(name)
    print("  accuracy:", round(accuracy_score(y_true, pred), 5),
          "recall:", round(recall_score(y_true, pred), 3),
          "precision:", round(precision_score(y_true, pred, zero_division=0), 3))
    print("  [[TN FP] [FN TP]] =", confusion_matrix(y_true, pred).tolist())


def cutoff_loop(name, proba, y_true, cutoffs=(0.5, 0.3, 0.1, 0.05)):
    """Try several cutoffs on a set of fraud scores."""
    print(name)
    for t in cutoffs:
        p = (proba >= t).astype(int)
        print(" ", t, "recall:", round(recall_score(y_true, p), 3),
              "precision:", round(precision_score(y_true, p, zero_division=0), 3))


# ---- Step 3: the do-nothing baseline ----
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
report("Dummy (always genuine)", y_test, dummy.predict(X_test))

# ---- Step 4: logistic regression (scaling needed, fit on train only) ----
lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
lr.fit(X_train, y_train)
report("Logistic regression", y_test, lr.predict(X_test))

# ---- Step 5: move the cutoff ----
proba_lr = lr.predict_proba(X_test)[:, 1]
cutoff_loop("Logistic regression cutoffs (test)", proba_lr, y_test)

# ---- Step 6: class_weight balanced (fraud rows count about 577x more) ----
lr_w = make_pipeline(StandardScaler(),
                     LogisticRegression(max_iter=1000, class_weight="balanced"))
lr_w.fit(X_train, y_train)
report("Logistic regression, balanced", y_test, lr_w.predict(X_test))

# ---- Step 7: XGBoost (no scaling needed) ----
xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                    random_state=42, eval_metric="logloss")
xgb.fit(X_train, y_train)
report("XGBoost", y_test, xgb.predict(X_test))

# ---- Step 8: XGBoost cutoffs ----
proba_x = xgb.predict_proba(X_test)[:, 1]
cutoff_loop("XGBoost cutoffs (test)", proba_x, y_test)

# ---- Step 9: average precision, one score for the whole curve ----
print("LogReg AP:", round(average_precision_score(y_test, proba_lr), 3))
print("XGBoost AP:", round(average_precision_score(y_test, proba_x), 3))

# ---- Step 10: scale_pos_weight (same idea as class_weight, for XGBoost) ----
ratio = (y_train == 0).sum() / (y_train == 1).sum()
print("ratio:", round(ratio, 1))
xgb_w = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                      random_state=42, eval_metric="logloss",
                      scale_pos_weight=ratio)
xgb_w.fit(X_train, y_train)
proba_w = xgb_w.predict_proba(X_test)[:, 1]
report("XGBoost, scale_pos_weight", y_test, (proba_w >= 0.5).astype(int))
print("XGBoost weighted AP:", round(average_precision_score(y_test, proba_w), 3))

# ---- Step 11: validation set, pick the cutoff without touching test ----
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train)
print("train:", X_tr.shape, "frauds:", y_tr.sum())
print("val:  ", X_val.shape, "frauds:", y_val.sum())
print("test: ", X_test.shape, "frauds:", y_test.sum())

xgb_v = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
                      random_state=42, eval_metric="logloss")
xgb_v.fit(X_tr, y_tr)                        # learn from train only
proba_v = xgb_v.predict_proba(X_val)[:, 1]   # score the validation rows
cutoff_loop("XGBoost cutoffs (validation)", proba_v, y_val)

# ---- Step 12 (later): retrain on all of X_train, use the chosen cutoff on test ONCE ----
# CUTOFF = 0.5   # chosen from the validation results above
# final = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.1,
#                       random_state=42, eval_metric="logloss").fit(X_train, y_train)
# report("FINAL on test", y_test, (final.predict_proba(X_test)[:, 1] >= CUTOFF).astype(int))