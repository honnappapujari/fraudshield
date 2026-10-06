import pandas as pd
import matplotlib
matplotlib.use("Agg")  # draw into a file instead of opening a window
import matplotlib.pyplot as plt

df = pd.read_csv("data/creditcard.csv")
counts = df["Class"].value_counts().sort_index()

plt.bar(["genuine (0)", "fraud (1)"], counts.values)
plt.title("Transactions by class")
plt.ylabel("count")
plt.savefig("outputs/class_counts.png", dpi=120)
print("saved outputs/class_counts.png")