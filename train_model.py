import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

# 1. Load dataset
df = pd.read_csv("vendor_data.csv")

# 2. Features and target
X = df[["daily_income", "daily_expense", "monthly_savings", "daily_upi_count"]]
y = df["repaid_loan"]

# 3. Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
preds = model.predict(X_test)
acc = accuracy_score(y_test, preds)
print("Training complete!")
print(f"Model Accuracy: {acc * 100:.2f}%")

# 6. Save model
with open("credit_model.pkl", "wb") as f:
    pickle.dump(model, f)
print("Saved trained model to 'credit_model.pkl'.")