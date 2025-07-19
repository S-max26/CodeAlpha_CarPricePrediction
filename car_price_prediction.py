# car_price_prediction_v2.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
data_path = "C:/Users/shash/OneDrive/Desktop/Internship 2/task 3/car data.csv"
car_data = pd.read_csv(data_path)

# Display basic dataset info
print("📊 Sample Data:\n", car_data.head())
print("\nℹ️ Dataset Info:\n")
car_data.info()
print("\n🧹 Null Values:\n", car_data.isnull().sum())

# Encode categorical features
encoder = LabelEncoder()
for col in ['Fuel_Type', 'Selling_type', 'Transmission']:
    car_data[col] = encoder.fit_transform(car_data[col])

# Select features and target
features = car_data[['Present_Price', 'Driven_kms', 'Fuel_Type', 'Selling_type', 'Transmission', 'Owner']]
target = car_data['Selling_Price']

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    features, target, test_size=0.2, random_state=42
)

# Train linear regression model
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Make predictions
predicted_prices = regressor.predict(X_test)

# Evaluate model
mse_score = mean_squared_error(y_test, predicted_prices)
r2 = r2_score(y_test, predicted_prices)

print(f"\n📉 Mean Squared Error: {mse_score:.2f}")
print(f"📈 R² Score: {r2:.2f}")

# Plot Actual vs Predicted values
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=predicted_prices, color='orange')
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.title("Actual vs Predicted Car Prices")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
