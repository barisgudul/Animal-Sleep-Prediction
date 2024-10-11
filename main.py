import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import warnings

# Suppress certain warnings for cleaner output
warnings.simplefilter(action="ignore", category=FutureWarning)

# Load the dataset
df = pd.read_csv("msleep.csv")

# Fill missing values in the "sleep_rem" column with the column's mean value
df["sleep_rem"].fillna(df["sleep_rem"].mean(), inplace=True)

# Fill missing values in the "brainwt" column with the column's mean value
df["brainwt"].fillna(df["brainwt"].mean(), inplace=True)

# Convert the "vore" column (e.g., carni, herbi) to numerical form using one-hot encoding
df = pd.get_dummies(df, columns=["vore"])

# Apply log transformation to weight data to make it more normally distributed
df['log_bodywt'] = np.log1p(df['bodywt'])
df['log_brainwt'] = np.log1p(df['brainwt'])

# New feature: Brain-to-body weight ratio (brain/body)
df['brain_body_ratio'] = df['brainwt'] / df['bodywt']

# Fill missing values in this ratio with the mean
df['brain_body_ratio'].fillna(df['brain_body_ratio'].mean(), inplace=True)

# Fill missing values in the "sleep_cycle" column with the mean
df["sleep_cycle"].fillna(df["sleep_cycle"].mean(), inplace=True)

# Convert the "conservation" and "order" columns to numerical form using one-hot encoding
df = pd.get_dummies(df, columns=["conservation", "order"], drop_first=True)

# Target variable: total sleep time ("sleep_total")
y = df["sleep_total"]

# Features: Selecting variables for the model
x = df[[
    "log_bodywt", "log_brainwt", "sleep_rem", "sleep_cycle", "vore_carni",
    "vore_herbi", "vore_omni", "vore_insecti", "brain_body_ratio"
] + [col for col in df.columns if 'conservation_' in col or 'order_' in col]]

# Scaling the features: Standardization (mean 0, standard deviation 1)
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)

# Splitting data into training and testing sets (80% training, 20% testing)
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

# Define the model: Creating a Random Forest Regressor
model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42)

# Train the model using the training data
model.fit(x_train, y_train)

# Make predictions on the test data
y_pred = model.predict(x_test)

# Evaluate the model's performance (MSE, MAE, and R²)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print performance metrics
print('Mean Squared Error (MSE):', mse)
print('Mean Absolute Error (MAE):', mae)
print('R-Squared (R²) Score:', r2)

# Outlier analysis: Identify the observation with the largest error
residuals = y_test - y_pred
outlier_index = residuals.abs().idxmax()  # Index of the largest error
outlier_pred_value = y_pred[residuals.abs().argmax()]  # Predicted value for this index
outlier_real_value = y_test.loc[outlier_index]  # Actual value for this index

# Find the feature values corresponding to this outlier
outlier_x_values = x_test[residuals.abs().argmax()]

# Print details of the outlier
print(f"Outlier index: {outlier_index}")
print(f"Outlier x values: {outlier_x_values}")
print(f"Predicted value: {outlier_pred_value}, Real value: {outlier_real_value}")

# Make predictions for the entire dataset
y_pred_all = model.predict(x_scaled)

# Add predicted sleep totals to the dataframe
df['predicted_sleep_total'] = y_pred_all

# Display the real and predicted sleep totals for each animal
results = df[['name', 'sleep_total', 'predicted_sleep_total']]
print("Animals' Actual and Predicted Sleep Totals:")
print(results)
