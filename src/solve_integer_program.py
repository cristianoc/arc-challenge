from typing import Any, List, Tuple, Optional
import pulp  # type: ignore
import random

from rule_based_selector import Features
from grid_data import logger

pulp: Any = pulp
random: Any = random

def find_weights_and_bias(samples: List[Features], goals: List[int], desc:str) -> Optional[Tuple[Features, int]]:
    """
    Finds integer weights and bias such that:
    
        Σ (W[feature] * f[feature]) + b = goal_i
    
    for each features dictionary `f` in `samples`.
    
    Args:
        samples: List of feature-value dictionaries.
        goals: List of target values.

    Returns:
        A tuple (W, b) where:
            W: Dictionary of weights.
            b: Integer bias.
        Returns None if no exact solution is found.
    """
    if not samples or len(samples) != len(goals):
        return None

    feature_names = list(samples[0].keys())
    num_samples = len(samples)

    # Create a linear programming problem
    problem = pulp.LpProblem("Integer_Programming_Example", pulp.LpMinimize)

    # Variables: W vector of integers (one for each feature) and b as an integer
    W = {feature: pulp.LpVariable(f"W_{feature}", lowBound=0, upBound=None, cat='Integer')
         for feature in feature_names}
    b = pulp.LpVariable("b", lowBound=0, upBound=None, cat='Integer')

    # Constraints: Σ (W[feature] * f[feature]) + b = goal_i for each features dictionary
    for i in range(num_samples):
        constraint = pulp.lpSum(samples[i][feature] * W[feature] for feature in feature_names) + b == goals[i]
        problem += constraint

    # Objective: Minimize the sum of W and bias
    problem += pulp.lpSum(W[feature] for feature in feature_names) + b
    

    # Solve the problem using the CBC solver with suppressed output
    problem.solve(pulp.PULP_CBC_CMD(msg=False))

    # Retrieve results
    W_solution = {feature: pulp.value(W[feature]) for feature in feature_names}
    b_solution = pulp.value(b)

    # Check if the solution is perfect
    is_perfect = True
    for i in range(num_samples):
        calculated_value = sum(samples[i][feature] * W_solution[feature] for feature in feature_names) + b_solution
        if calculated_value != goals[i]:
            is_perfect = False
            break

    logger.debug(f"Solution {desc}: Weights: {W_solution}, Bias: {b_solution}")

    # Check if the solution is plausible
    plausible = True
    # bias must be an integer between 0 and 2
    if b_solution % 1 != 0 or b_solution < 0 or b_solution > 2:
        logger.info("Bias is not an integer between 0 and 2")
        plausible = False
    # weights must be integers unless there's only one weight which is 1/2 or 1/3 and the bias is 0
    if len(W_solution) > 1:
        for weight in W_solution.values():
            if weight % 1 != 0:
                if weight != 0.5 and weight != 1/3:
                    logger.info(f"One of the weights is not an integer: {weight} and not 1/2 or 1/3")
                    plausible = False
                    break
    # weights must be integers and bias must be nonnegative integer and at most one weight can be nonzero
    else:
        if b_solution % 1 != 0 or b_solution < 0 or sum(W_solution.values()) > 1:
            plausible = False

    if is_perfect and plausible:
        return W_solution, b_solution
    else:
        return None

# Test function
def test_find_weights_and_bias():
    random.seed(42)
    
    # Parameters
    num_samples = 10
    num_features = 5

    # Generate random features and corresponding goals
    feature_names = [f"feature_{i}" for i in range(num_features)]
    samples : List[Features] = [
        {feature: random.randint(1, 10) for feature in feature_names} 
        for _ in range(num_samples)
    ]
    goals = [2 * samples[i]["feature_0"] + samples[i]["feature_1"] for i in range(num_samples)]

    # Call the library function
    result = find_weights_and_bias(samples, goals, "Test")

    # Output the generated data, solution, and check result
    logger.info("Samples:")
    for sample in samples:
        logger.info(sample)

    if result:
        weights, bias = result
        logger.info(f"\nWeights:{weights}")
        logger.info(f"Bias:{bias}")
        logger.info("\nThe solution perfectly satisfies all the constraints!")
    else:
        logger.info("\nNo exact solution was found.")

# Run the test function
if __name__ == "__main__":
    test_find_weights_and_bias()
