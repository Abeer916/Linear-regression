# Import necessary libraries
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load the dataset
iris_df = pd.read_csv("E:\Python projects\Iris dataset linear regression\Iris.csv")

# Define the feature (X) and target (y)
X = iris_df[['SepalLengthCm']]  # Predictor (Sepal Length)
y = iris_df['PetalLengthCm']     # Target (Petal Length)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R^2 Score: {r2}")

# Plot the results
plt.scatter(X_test, y_test, color='blue', label='Actual Petal Length')
plt.plot(X_test, y_pred, color='red', label='Predicted Petal Length')
plt.xlabel('Sepal Length (cm)')
plt.ylabel('Petal Length (cm)')
plt.legend()
plt.title('Simple Linear Regression on Iris Dataset')
plt.show()
