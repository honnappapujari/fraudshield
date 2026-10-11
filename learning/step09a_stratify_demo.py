from sklearn.model_selection import train_test_split

X = list(range(20))
y = [0] * 16 + [1] * 4

for seed in [1, 2, 3, 4, 5]:
    _, _, _, y_test = train_test_split(X, y, test_size=0.25, random_state=seed)
    print("no stratify, seed", seed, "-> frauds in test:", sum(y_test))

print()
for seed in [1, 2, 3, 4, 5]:
    _, _, _, y_test = train_test_split(X, y, test_size=0.25, random_state=seed, stratify=y)
    print("stratify,    seed", seed, "-> frauds in test:", sum(y_test))