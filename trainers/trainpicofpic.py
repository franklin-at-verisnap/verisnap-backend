from sklearn.linear_model import LogisticRegression
import joblib
import pandas as pd

# Example: load your CSV of collected examples
df = pd.read_csv("../test/picofpicdataset.csv")
X = df[['planar_ratio','variance','edge_density','border_edge_density']]
y = df['label']

clf = LogisticRegression()
clf.fit(X, y)

# Evaluate quickly
print("Train accuracy:", clf.score(X,y))

# Save for inference
joblib.dump(clf, "../models/p2p_classifier.joblib")
