from typing import Any, List, Tuple, Optional
import pulp  # type: ignore
import random

from rule_based_selector import Features

pulp: Any = pulp
random: Any = random

def find_weights_and_bias(samples: List[Features], goals: List[int]) -> Optional[Tuple[Features, int]]:
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

    # Objective: Minimize the sum of W
    problem += pulp.lpSum(W[feature] for feature in feature_names)

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

    if is_perfect:
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
    samples = [
        {feature: random.randint(1, 10) for feature in feature_names} 
        for _ in range(num_samples)
    ]
    goals = [2 * samples[i]["feature_0"] + samples[i]["feature_1"] for i in range(num_samples)]

    # Call the library function
    result = find_weights_and_bias(samples, goals)

    # Output the generated data, solution, and check result
    print("Samples:")
    for sample in samples:
        print(sample)

    if result:
        weights, bias = result
        print("\nWeights:", weights)
        print("Bias:", bias)
        print("\nThe solution perfectly satisfies all the constraints!")
    else:
        print("\nNo exact solution was found.")

# Run the test function
if __name__ == "__main__":
    test_find_weights_and_bias()
