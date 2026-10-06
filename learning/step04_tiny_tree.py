from sklearn.tree import DecisionTreeClassifier, export_text

# Each row: [amount in rupees, hour of day (0-23)]
X = [
    [500, 14], [1200, 11], [300, 16], [800, 10],
    [2000, 15], [150, 18], [700, 12], [60, 20],
    [9000, 3], [15000, 2], [8000, 4], [12000, 1],
]
# Answers: 0 = genuine, 1 = fraud
y = [0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1]

model = DecisionTreeClassifier(max_depth=2, random_state=42)
model.fit(X, y)

print(export_text(model, feature_names=["amount", "hour"]))

new_transactions = [[10000, 3], [400, 13], [9000, 14]]
print(model.predict(new_transactions))
print(model.predict_proba(new_transactions))
print("Accuracy on the 12 training rows:", model.score(X, y))