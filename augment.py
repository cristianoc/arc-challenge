import numpy as np
from sklearn.linear_model import LogisticRegression

# Augmentation parameters
n_values = [2, 3, 4]

# Generate a larger training dataset
x_large_train = np.arange(0, 1000)
X_large_augmented = np.hstack([x_large_train.reshape(-1, 1) % n for n in n_values])
y_large_mod_3 = x_large_train % 3

# Train the logistic regression model for x % 3 with more data
logistic_large_mod_3 = LogisticRegression(multi_class='multinomial', max_iter=1000, solver='lbfgs')
logistic_large_mod_3.fit(X_large_augmented, y_large_mod_3)

# Generate test data far outside the training range
x_far_test = np.arange(10000, 10100)
X_far_test_augmented = np.hstack([x_far_test.reshape(-1, 1) % n for n in n_values])
y_far_test = x_far_test % 3

# Evaluate the logistic regression model for x % 3 on far away test data
probs_far_mod_3 = logistic_large_mod_3.predict_proba(X_far_test_augmented)
far_test_accuracy_mod_3 = logistic_large_mod_3.score(X_far_test_augmented, y_far_test)

# Calculate confidence levels for the far away test data
confidence_levels_far_mod_3 = []
for prob in probs_far_mod_3:
    max_prob = max(prob)
    if max_prob >= 0.8:
        confidence_levels_far_mod_3.append("High")
    elif max_prob >= 0.6:
        confidence_levels_far_mod_3.append("Medium")
    else:
        confidence_levels_far_mod_3.append("Low")

# Display the test accuracy and the confidence levels for the first few instances
print("Test Accuracy for x % 3:", far_test_accuracy_mod_3)
print("Prediction Probabilities for first 5 instances:")
print(probs_far_mod_3[:5])
print("Confidence Levels for first 5 instances:")
print(confidence_levels_far_mod_3[:5])
