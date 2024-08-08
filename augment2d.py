from matplotlib import colors
import numpy as np
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from typing import List, Tuple, Callable, Union

# Type aliases for readability
ArrayLike = Union[np.ndarray, List[float]]
FeatureFunction = Callable[[ArrayLike, ArrayLike, int], ArrayLike]

# Augmentation parameters
modulo_n_values: List[int] = [2, 3, 4]

# Grid pattern generation
def generate_grid(size: int, modulo: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(size)
    y = np.arange(size)
    X, Y = np.meshgrid(x, y)
    color_indices = (X + Y) % modulo  # Checkerboard pattern
    return X, Y, color_indices

# List of operations to be applied for augmentation and their corresponding names
operations: List[Tuple[FeatureFunction, str]] = [
    (lambda x, y, n: x % n, "x % {n}"),
    (lambda x, y, n: y % n, "y % {n}"),
    (lambda x, y, n: (x + y) % n, "(x + y) % {n}"),
    (lambda x, y, n: (x - y + n) % n, "(x - y + {n}) % {n}"),
    (lambda x, y, n: (y - x + n) % n, "(y - x + {n}) % {n}")
]

# Feature augmentation
def augment_features(X: np.ndarray, Y: np.ndarray, n_values: List[int]) -> np.ndarray:
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    
    features: List[np.ndarray] = []
    for op, _ in operations:
        for n in n_values:
            features.append(op(X_flat, Y_flat, n))
    
    return np.column_stack(features)

# Generate feature function names
def generate_feature_functions(n_values: List[int]) -> List[str]:
    feature_functions: List[str] = []
    for _, name_template in operations:
        for n in n_values:
            feature_functions.append(name_template.format(n=n))
    return feature_functions

# Train logistic regression model
def train_model(X: np.ndarray, y: np.ndarray) -> LogisticRegression:
    model = LogisticRegression(
        multi_class='multinomial',
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X, y)
    return model

# Evaluate the model
def evaluate_model(model: LogisticRegression, X: np.ndarray, y: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray, List[str]]:
    probs = model.predict_proba(X)
    accuracy = model.score(X, y)
    predictions = model.predict(X)
    confidence_levels: List[str] = []
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
def print_model_coefficients(model: LogisticRegression, feature_functions: List[str]) -> None:
    coefficients = model.coef_[0]
    intercept = model.intercept_[0]
    print("Learned function:")
    for coef, func in zip(coefficients, feature_functions):
        rounded_coef = round(coef, 2)
        if rounded_coef != 0:
            print(f"{rounded_coef} * {func}")
    print("Intercept:", round(intercept, 2))

# Derive the most likely predicted function
def derive_predicted_function(model: LogisticRegression, feature_functions: List[str]) -> str:
    coefficients = model.coef_[0]
    
    # Identify the most significant term(s) (highest absolute value of coefficients)
    significant_terms = sorted(zip(coefficients, feature_functions), key=lambda x: abs(x[0]), reverse=True)
    
    # Get the top significant term
    top_term = significant_terms[0][1]
    
    # Construct the function
    predicted_function = f"f(x, y) => {top_term}"
    
    return predicted_function

# Simplified predicted class function based on significant terms
def predicted_class(x, y):
    term1 = (x + y) % 4

    # Considering the intercept and most significant terms
    if term1 in [0, 2]:
        return 0
    else:
        return 1

# Define the custom color scheme as a list of colors
color_scheme = [
    '#000000',  # black
    '#0074D9',  # blue
    '#FF4136',  # red
    '#2ECC40',  # green
    '#FFDC00',  # yellow
    '#AAAAAA',  # grey
    '#F012BE',  # fuschia
    '#FF851B',  # orange
    '#870C25',   # brown
    '#7FDBFF',  # teal
]

cmap = colors.ListedColormap(color_scheme)


# Visualization
def visualize_results(X: np.ndarray, true_color_indices: np.ndarray, predicted_color_indices: np.ndarray, grid_size: int) -> None:
    true_grid = true_color_indices.reshape(grid_size, grid_size)
    predicted_grid = predicted_color_indices.reshape(grid_size, grid_size)
    
    plt.figure(figsize=(12, 5))

    # Plot the expected checkerboard pattern with colors
    plt.subplot(1, 2, 1)
    plt.title("Expected Checkerboard Pattern")
    plt.imshow(true_grid, cmap=cmap)
    plt.colorbar(label='Color Index')

    # Plot the predicted checkerboard pattern with colors
    plt.subplot(1, 2, 2)
    plt.title("Predicted Checkerboard Pattern")
    plt.imshow(predicted_grid, cmap=cmap)
    plt.colorbar(label='Color Index')

    plt.tight_layout()
    plt.show()

# Main function
def main() -> None:
    # Generate training data (checkerboard pattern)
    grid_size = 100
    modulo = 4
    X_train, Y_train, color_indices_train = generate_grid(grid_size, modulo)
    X_augmented_train = augment_features(X_train, Y_train, modulo_n_values)
    
    # Train the logistic regression model
    logistic_model = train_model(X_augmented_train, color_indices_train.flatten())
    
    # Generate feature functions
    feature_functions = generate_feature_functions(modulo_n_values)
    
    # Print the model coefficients
    print_model_coefficients(logistic_model, feature_functions)
    
    # Derive and print the most likely predicted function
    predicted_function = derive_predicted_function(logistic_model, feature_functions)
    print("The most likely predicted function:")
    print(predicted_function)
    
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
