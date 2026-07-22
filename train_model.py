import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

# ==========================
# LOAD DATASET
# ==========================
df = pd.read_csv("superstore/superstore.csv", encoding="latin1")

# Remove unwanted column
df = df.drop(columns=["è®°å½æ°"], errors="ignore")

# Remove duplicates
df = df.drop_duplicates()

# Convert date columns
df["Order.Date"] = pd.to_datetime(df["Order.Date"])
df["Ship.Date"] = pd.to_datetime(df["Ship.Date"])

# Create new features
df["Order_Year"] = df["Order.Date"].dt.year
df["Order_Month"] = df["Order.Date"].dt.month
df["Order_Day"] = df["Order.Date"].dt.day

# ==========================
# SELECT FEATURES
# ==========================
features = [
    "Category",
    "Sub.Category",
    "Quantity",
    "Discount",
    "Shipping.Cost",
    "Market2",
    "Segment"
]

target = "Sales"

# ==========================
# LABEL ENCODING
# ==========================
# ==========================
# LABEL ENCODING
# ==========================
from sklearn.preprocessing import LabelEncoder

encoders = {}

for col in features:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    encoders[col] = le

# ==========================
# TRAIN TEST SPLIT
# ==========================
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# ==========================
# TRAIN MODEL
# ==========================
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

# ==========================
# PREDICTIONS
# ==========================
y_pred = model.predict(X_test)

# ==========================
# EVALUATION
# ==========================
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\n========== MODEL PERFORMANCE ==========")
print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# ==========================
# SAVE MODEL
# ==========================
joblib.dump(model, "sales_prediction_model.pkl")
joblib.dump(encoders, "label_encoders.pkl")

print("\nModel saved successfully!")
print("sales_prediction_model.pkl")
print("label_encoders.pkl")