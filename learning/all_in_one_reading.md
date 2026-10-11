# Fraud detection: how we improved the model, step by step

Dataset: 284,807 credit card transactions, only 492 are fraud (about 1 in 578).
All results below come from the same hidden test set (56,962 rows, 98 frauds).

## The journey

| # | What we did | Frauds caught (of 98) | False alarms | What we learned |
|---|---|---|---|---|
| 1 | Do-nothing model (always "genuine") | 0 | 0 | 99.83% accuracy but useless. Accuracy lies on rare events. |
| 2 | Logistic regression (with StandardScaler) | 62 | 13 | First real model. Recall 0.633, precision 0.827. |
| 3 | Logistic regression, cutoff 0.05 | about 81 | about 43 | Lower cutoff catches more fraud but adds false alarms. |
| 4 | Logistic regression, class_weight="balanced" | 90 | 1,386 | Catches more but far too many alarms. Same trade-off curve. |
| 5 | XGBoost | 78 | 8 | Biggest real improvement: far fewer false alarms. |
| 6 | XGBoost, cutoff 0.05 | about 82 | about 39 | Small gain only. |
| 7 | Average precision: LogReg 0.741, XGBoost 0.832 | - | - | One score for the whole precision-recall curve. XGBoost is clearly better. |
| 8 | XGBoost with scale_pos_weight (about 577) | 82 | about 49 | Same trade-off as lowering the cutoff. Nothing new. |
| 9 | Validation set to pick the cutoff honestly | - | - | On validation the cutoff barely mattered for XGBoost. |

## Key ideas in plain words

1. **Accuracy lies on rare events.** Use recall and precision.
2. **Recall** = frauds caught / all real frauds. "How much fraud did we find?"
3. **Precision** = real frauds / everything we flagged. "How many alerts were true?"
4. **Cutoff** = the score a transaction must reach to be flagged.
   Lower cutoff: more alerts, recall up, precision down.
   Higher cutoff: fewer alerts, recall down, precision up.
5. **Class weights do the same as lowering the cutoff.** They slide along the same curve.
6. **A better model matters most.** Switching to XGBoost was the only real jump.
7. **Honest testing.** Keep a hidden test set. Pick the cutoff on a validation set, not on test.
8. **Ceiling.** About 16 to 20 frauds look completely genuine to the model.

## Full code (runs top to bottom in Colab)

```python
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
URL = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"
df = pd.read_csv(URL)
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
```

## Parked for later

- Hour-of-day feature from the `Time` column.
- XGBoost tuning (depth, number of trees, learning rate).
- Step 12 above: the final honest test score.
- SHAP: why the model flags a transaction.
- Cross-validation for a steadier cutoff.