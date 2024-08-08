import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Augmentation parameters
modulo_n_values = [2, 3, 4]

# Grid pattern generation
def generate_grid(size, modulo):
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    color_indices = (X + Y) % modulo  # Checkerboard pattern
    return X, Y, color_indices

# List of operations to be applied for augmentation and their corresponding names
operations = [
    (lambda x, y, n: x % n, "x % {n}"),
    (lambda x, y, n: y % n, "y % {n}"),
    (lambda x, y, n: (x + y) % n, "(x + y) % {n}"),
    (lambda x, y, n: (x - y + n) % n, "(x - y + {n}) % {n}"),
    (lambda x, y, n: (y - x + n) % n, "(y - x + {n}) % {n}")
]

# Feature augmentation
def augment_features(X, Y, n_values):
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    features = []
    for op, _ in operations:
        for n in n_values:
            features.append(op(X_flat, Y_flat, n))
    
    return np.column_stack(features)

# Generate feature function names
def generate_feature_functions(n_values):
    feature_functions = []
    for _, name_template in operations:
        for n in n_values:
            feature_functions.append(name_template.format(n=n))
    return feature_functions

# Train logistic regression model
def train_model(X, y):
    model = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X, y)
    return model

# Evaluate the model
def evaluate_model(model, X, y):
    probs = model.predict_proba(X)
    accuracy = model.score(X, y)
    predictions = model.predict(X)
    confidence_levels = []
    for prob in probs:
        max_prob = max(prob)
        if max_prob >= 0.8:
            confidence_levels.append("High")
        elif max_prob >= 0.6:
            confidence_levels.append("Medium")
        else:
            confidence_levels.append("Low")
    return accuracy, probs, predictions, confidence_levels

# Print model coefficients
def print_model_coefficients(model, feature_functions):
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    print("Learned function:")
    for coef, func in zip(coefficients, feature_functions):
        rounded_coef = round(coef, 2)
        if rounded_coef != 0:
            print(f"{rounded_coef} * {func}")
    print("Intercept:", round(intercept, 2))

# Visualization
def visualize_results(X, true_color_indices, predicted_color_indices, grid_size):
    true_grid = true_color_indices.reshape(grid_size, grid_size)
    predicted_grid = predicted_color_indices.reshape(grid_size, grid_size)
    
    plt.figure(figsize=(12, 5))

    # Plot the expected checkerboard pattern
    plt.subplot(1, 2, 1)
    plt.title("Expected Checkerboard Pattern")
    plt.imshow(true_grid, cmap='gray')
    plt.colorbar(label='Color Index')

    # Plot the predicted checkerboard pattern
    plt.subplot(1, 2, 2)
    plt.title("Predicted Checkerboard Pattern")
    plt.imshow(predicted_grid, cmap='gray')
    plt.colorbar(label='Color Index')

    plt.tight_layout()
    plt.show()

# Main function
def main():
    # Generate training data (checkerboard pattern)
    grid_size = 100
    modulo = 3
    X_train, Y_train, color_indices_train = generate_grid(grid_size, modulo)
    X_augmented_train = augment_features(X_train, Y_train, modulo_n_values)
    
    # Train the logistic regression model
    logistic_model = train_model(X_augmented_train, color_indices_train.flatten())
    
    # Generate feature functions
    feature_functions = generate_feature_functions(modulo_n_values)
    
    # Print the model coefficients
    print_model_coefficients(logistic_model, feature_functions)
    
    # Generate test data (different range to test generalization)
    test_grid_size = 100
    X_test, Y_test, color_indices_test = generate_grid(test_grid_size, modulo)
    X_augmented_test = augment_features(X_test, Y_test, modulo_n_values)
    
    # Evaluate the logistic regression model
    test_accuracy, probs_test, predictions, confidence_levels_test = evaluate_model(logistic_model, X_augmented_test, color_indices_test.flatten())
    
    # Display the test accuracy and the confidence levels for the first few instances
    print("Test Accuracy for checkerboard pattern:", test_accuracy)
    print("Prediction Probabilities for first 5 instances:")
    print(probs_test[:5])
    print("Confidence Levels for first 5 instances:")
    print(confidence_levels_test[:5])
    
    # Visualize the results
    visualize_results(X_test, color_indices_test.flatten(), predictions, test_grid_size)

if __name__ == "__main__":
    main()
