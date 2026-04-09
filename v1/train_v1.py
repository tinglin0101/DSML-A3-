import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
# import os

# 1. Load data
# bath_path = ""
# print(os.getcwd(),"------")

train_df = pd.read_csv("train_2022.csv")
test_df = pd.read_csv("test_no_answer_2022.csv")

# print(f"Train shape: {train_df.shape}")
# print(f"Test shape: {test_df.shape}")
# print(f"Label distribution:\n{train_df['LABEL'].value_counts()}")

# 2. Feature extraction with TF-IDF
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2), sublinear_tf=True)
X_train = tfidf.fit_transform(train_df["TEXT"])
X_test = tfidf.transform(test_df["TEXT"])
y_train = train_df["LABEL"]

# 3. Train Logistic Regression
model = LogisticRegression(max_iter=1000, C=1.0)

# Cross-validation
scores = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy")
print(f"\nCross-validation accuracy: {scores.mean():.4f} (+/- {scores.std():.4f})")

# Fit on full training set
model.fit(X_train, y_train)

# 4. Predict on test set
predictions = model.predict(X_test)

# 5. Save results
output_df = pd.DataFrame({
    "row_id": test_df["row_id"],
    "label": predictions
})
output_df.to_csv("v1.csv", index=False)
print(f"\nPredictions saved to v1.csv ({len(output_df)} rows)")
print(f"Prediction distribution:\n{pd.Series(predictions).value_counts()}")
