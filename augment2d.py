import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Augmentation parameters
n_values = [2, 3, 4]

# Checkerboard pattern generation
def generate_checkerboard(size):
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    color_indices = (X + Y) % 2  # Checkerboard pattern
    return X, Y, color_indices

# Generate training data (checkerboard pattern)
grid_size = 100
X_train, Y_train, color_indices_train = generate_checkerboard(grid_size)

# Flatten the training data for model input
X_train_flat = X_train.flatten()
Y_train_flat = Y_train.flatten()
color_indices_train_flat = color_indices_train.flatten()

# Create augmented features using modulo operations
xmod = [X_train_flat % n for n in n_values]
ymod = [Y_train_flat % n for n in n_values]
xy_sum_mod = [(X_train_flat + Y_train_flat) % n for n in n_values]
diag1_mod = [(X_train_flat - Y_train_flat + n) % n for n in n_values]
diag2_mod = [(Y_train_flat - X_train_flat + n) % n for n in n_values]
X_augmented_train = np.column_stack(xmod + ymod + xy_sum_mod + diag1_mod + diag2_mod)

# Train the logistic regression model
logistic_model = LogisticRegression(
    multi_class='multinomial',
    max_iter=1000,
    solver='lbfgs'
)
logistic_model.fit(X_augmented_train, color_indices_train_flat)

# Extract coefficients to express as a linear function
coefficients = logistic_model.coef_[0]
intercept = logistic_model.intercept_[0]

# Define the mapping of feature indices to their corresponding functions
feature_functions = [
    "x % 2", "x % 3", "x % 4", 
    "y % 2", "y % 3", "y % 4", 
    "(x + y) % 2", "(x + y) % 3", "(x + y) % 4", 
    "(x - y + 2) % 2", "(x - y + 3) % 3", "(x - y + 4) % 4", 
    "(y - x + 2) % 2", "(y - x + 3) % 3", "(y - x + 4) % 4"
]

# Print the rounded coefficients and corresponding functions
print("Learned function:")
for coef, func in zip(coefficients, feature_functions):
    rounded_coef = round(coef, 2)  # Round the coefficients to 2 decimal places
    if rounded_coef != 0:
        print(f"{rounded_coef} * {func}")

print("Intercept:", round(intercept, 2))

# Generate test data (different range to test generalization)
test_grid_size = 100
x_test = np.arange(1000, 1000 + test_grid_size)
y_test = np.arange(1000, 1000 + test_grid_size)
X_test, Y_test = np.meshgrid(x_test, y_test)

# Flatten the test data for model input
X_test_flat = X_test.flatten()
Y_test_flat = Y_test.flatten()

# Create augmented features for test data
xmod = [X_test_flat % n for n in n_values]
ymod = [Y_test_flat % n for n in n_values]
xy_sum_mod = [(X_test_flat + Y_test_flat) % n for n in n_values]
diag1_mod = [(X_test_flat - Y_test_flat + n) % n for n in n_values]
diag2_mod = [(Y_test_flat - X_test_flat + n) % n for n in n_values]
X_augmented_test = np.column_stack(xmod + ymod + xy_sum_mod + diag1_mod + diag2_mod)

# Define the expected color index pattern for test data (checkerboard)
color_indices_test_flat = (X_test_flat + Y_test_flat) % 2

# Evaluate the logistic regression model
probs_test = logistic_model.predict_proba(X_augmented_test)
test_accuracy = logistic_model.score(X_augmented_test, color_indices_test_flat)
predictions = logistic_model.predict(X_augmented_test)

# Calculate confidence levels for the test data
confidence_levels_test = []
for prob in probs_test:
    max_prob = max(prob)
    if max_prob >= 0.8:
        confidence_levels_test.append("High")
    elif max_prob >= 0.6:
        confidence_levels_test.append("Medium")
    else:
        confidence_levels_test.append("Low")

# Display the test accuracy and the confidence levels for the first few instances
print("Test Accuracy for checkerboard pattern:", test_accuracy)
print("Prediction Probabilities for first 5 instances:")
print(probs_test[:5])
print("Confidence Levels for first 5 instances:")
print(confidence_levels_test[:5])

# Visualize the results
predictions_grid = predictions.reshape(test_grid_size, test_grid_size)

plt.figure(figsize=(12, 5))

# Plot the expected checkerboard pattern
plt.subplot(1, 2, 1)
plt.title("Expected Checkerboard Pattern")
plt.imshow(color_indices_test_flat.reshape(test_grid_size, test_grid_size), cmap='gray')
plt.colorbar(label='Color Index')

# Plot the predicted checkerboard pattern
plt.subplot(1, 2, 2)
plt.title("Predicted Checkerboard Pattern")
plt.imshow(predictions_grid, cmap='gray')
plt.colorbar(label='Color Index')

plt.tight_layout()
plt.show()
