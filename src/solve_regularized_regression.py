import logging
from typing import Any, List, Tuple, Optional
import pulp  # type: ignore
import random

from rule_based_selector import Features
from grid_data import logger

pulp: Any = pulp
random: Any = random

# Suppress the verbose output from the pulp solver
logging.getLogger('pulp').setLevel(logging.WARNING)

def solve_regularized_regression(features: List[Features], targets: List[int], description: str) -> Optional[Tuple[Features, int]]:
    """
    Solves for integer weights and bias that minimize the regularized objective:
    
        minimize ||W||_1 + |b|
    
    subject to:
    
        Σ (W[f] * x[f]) + b = y_i
    
    for each feature vector `x` in `features`.
    
    Args:
        features: List of feature-value dictionaries.
        targets: List of target values.

    Returns:
        A tuple (W, b) where:
            W: Dictionary of weights.
            b: Integer bias.
        Returns None if no feasible solution is found after applying the regularization constraints.
    """
    if not features or len(features) != len(targets):
        return None

    # find feature names as keys present in all features
    feature_names : List[str] = list(set.intersection(*[set(f.keys()) for f in features])) # type: ignore
    
    num_samples = len(features)

    # Define the linear programming problem with an objective to minimize (L1 regularization)
    optimization_problem = pulp.LpProblem("Regularized_Linear_Regression", pulp.LpMinimize)

    # Define the weight vector W (one variable per feature) and bias term b
    weights = {feature: pulp.LpVariable(f"W_{feature}", lowBound=0, upBound=None, cat='Integer')
               for feature in feature_names}
    bias = pulp.LpVariable("bias", lowBound=0, upBound=None, cat='Integer')

    # Add equality constraints for each sample
    for i in range(num_samples):
        constraint = pulp.lpSum(features[i][feature] * weights[feature] for feature in feature_names) + bias == targets[i]
        optimization_problem += constraint

    # Objective function: minimize L1 norm of weights and bias (regularization term)
    optimization_problem += pulp.lpSum(weights[feature] for feature in feature_names) + bias

    # Solve the optimization problem using the CBC solver with suppressed output
    optimization_problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Extract the optimized weights and bias
    optimized_weights = {feature: pulp.value(weights[feature]) for feature in feature_names}
    optimized_bias = pulp.value(bias)

    # Verify if the solution perfectly fits the data
    is_perfect_solution = all(
        sum(features[i][feature] * optimized_weights[feature] for feature in feature_names) + optimized_bias == targets[i]
        for i in range(num_samples)
    )

    logger.debug(f"Solution {description}: Weights: {optimized_weights}, Bias: {optimized_bias}")

    # Apply post-optimization regularization checks
    if is_perfect_solution and is_regularized_solution(optimized_weights, optimized_bias):
        return optimized_weights, optimized_bias
    else:
        return None

def is_regularized_solution(weights: Features, bias: int) -> bool:
    """
    Post-optimization regularization: Enforce additional constraints on the solution to ensure
    simplicity and generalization.

    Args:
        weights: Dictionary of optimized weights.
        bias: Optimized bias.

    Returns:
        True if the solution meets all regularization constraints, False otherwise.
    """

    # Regularization Constraint 1: Bias must be an integer between 0 and 2
    if bias % 1 != 0 or bias < 0 or bias > 2:
        logger.info("Bias is not an integer between 0 and 2")
        return False

    # Regularization Constraint 2: Weights must be integers unless special cases (e.g., single weight scenario)
    if len(weights) > 1:
        for weight in weights.values():
            if weight % 1 != 0 and weight not in [0.5, 1/3]:
                logger.info(f"One of the weights is not an integer: {weight} and not 1/2 or 1/3")
                return False
    # Regularization Constraint 3: Weights and bias checks for specific scenarios
    else:
        if bias % 1 != 0 or bias < 0 or sum(weights.values()) > 1:
            return False

    return True

# Test function for the regularized linear regression solver
def test_solve_regularized_regression():
    random.seed(42)
    
    # Parameters
    num_samples = 10
    num_features = 5

    # Generate random feature vectors and corresponding target values
    feature_names = [f"feature_{i}" for i in range(num_features)]
    features: List[Features] = [
        {feature: random.randint(1, 10) for feature in feature_names} 
        for _ in range(num_samples)
    ]
    targets = [2 * features[i]["feature_0"] + features[i]["feature_1"] for i in range(num_samples)]

    # Call the regularized regression solver
    result = solve_regularized_regression(features, targets, "Test")

    # Output the generated data, solution, and check result
    logger.info("Feature Vectors:")
    for feature in features:
        logger.info(feature)

    if result:
        weights, bias = result
        logger.info(f"\nOptimized Weights: {weights}")
        logger.info(f"Optimized Bias: {bias}")
        logger.info("\nThe solution perfectly satisfies all the constraints!")
    else:
        logger.info("\nNo exact solution was found.")

# Run the test function
if __name__ == "__main__":
    test_solve_regularized_regression()
