from typing import List, Dict, Optional

# Type aliases for clarity
Vector = List[int]

class DecisionRule:
    def __init__(self, correct_object: str, unique_features: List[int], correct_vector: Vector):
        self.correct_object = correct_object
        self.unique_features = unique_features
        self.correct_vector = correct_vector

    def __str__(self) -> str:
        if not self.unique_features:
            return f"No unique selection is possible for {self.correct_object}."
        
        conditions = [f"Feature {i+1} = {self.correct_vector[i]}" for i in self.unique_features]
        rule = " AND ".join(conditions)
        return f"Select {self.correct_object} if {rule}."

    def evaluate(self, vector: Vector) -> bool:
        """
        Evaluate if the given vector satisfies the decision rule.
        """
        if not self.unique_features:
            print(f"No decision rule available for {self.correct_object}.")
            return False
        return all(vector[i] == self.correct_vector[i] for i in self.unique_features)

def find_minimal_unique_features(correct_vector: Vector, other_vectors: List[Vector]) -> List[int]:
    """
    Identify the minimal set of unique features that differentiate the correct object from all other objects.
    Returns the indices of the minimal features that are necessary to uniquely identify the correct object.
    """
    all_features = [i for i in range(len(correct_vector)) if any(correct_vector[i] != other[i] for other in other_vectors)]
    minimal_features = all_features.copy()
    
    for feature in all_features:
        temp_features = [f for f in minimal_features if f != feature]
        temp_correct_vector = [correct_vector[f] for f in temp_features]
        temp_other_vectors = [[other[f] for f in temp_features] for other in other_vectors]
        
        # Check if the remaining features still uniquely identify the object
        if all(temp_correct_vector != other for other in temp_other_vectors):
            minimal_features.remove(feature)
    
    return minimal_features

def generate_decision_rule(correct_object: str, correct_vector: Vector, unique_features: List[int]) -> Optional[DecisionRule]:
    """
    Generate the decision rule based on unique features.
    """
    if not unique_features:
        return None
    return DecisionRule(correct_object, unique_features, correct_vector)

def select_object_minimal(experiment: Dict[str, Vector], correct_object_index: int) -> Optional[DecisionRule]:
    """
    Main function to process the experiment and return the minimal selection rule for a given correct object index.
    """
    object_names = list(experiment.keys())
    correct_object = object_names[correct_object_index]
    correct_vector = list(experiment.values())[correct_object_index]
    
    other_vectors = [v for i, v in enumerate(experiment.values()) if i != correct_object_index]
    
    minimal_unique_features = find_minimal_unique_features(correct_vector, other_vectors)
    decision_rule = generate_decision_rule(correct_object, correct_vector, minimal_unique_features)
    
    return decision_rule

# Testing the functions with the scenarios

def test():
    experiment_1 = {
        "Object A": [1, 1, 1, 0, 1, 1, 0],
        "Object B": [0, 1, 1, 0, 1, 0, 0],
        "Object C": [1, 0, 0, 1, 1, 1, 0],
    }

    experiment_2 = {
        "Object A": [1, 0, 1, 0, 1, 1, 0],
        "Object B": [1, 0, 1, 0, 1, 1, 0],
        "Object C": [1, 0, 1, 0, 1, 1, 0],
    }

    experiment_3 = {
        "Object A": [1, 0, 1, 1, 0, 0, 1],
        "Object B": [0, 1, 0, 0, 1, 1, 0],
        "Object C": [0, 0, 1, 0, 1, 0, 1],
    }

    experiments = {
        "Experiment 1": experiment_1,
        "Experiment 2": experiment_2,
        "Experiment 3": experiment_3,
    }

    results_minimal_with_indices = {
        "Experiment 1 (Object A)": select_object_minimal(experiment_1, 0),
        "Experiment 1 (Object C)": select_object_minimal(experiment_1, 2),
        "Experiment 2 (Object A)": select_object_minimal(experiment_2, 0),
        "Experiment 3 (Object A)": select_object_minimal(experiment_3, 0),
    }

    for r, decision_rule in results_minimal_with_indices.items():
        if decision_rule is None:
            print(f"{r}: No unique selection is possible.")
        else:
            print(f"{r}: {decision_rule}")
            # Test the decision rule on the correct vector
            experiment_name = r.split(" ")[0] + " " + r.split(" ")[1]  # Extract full experiment name like "Experiment 1"
            correct_object_name = r.split("(")[1][:-1]  # Extracts the object name from "Object A)" format
            correct_vector = experiments[experiment_name][correct_object_name]
            print(f"Evaluation result: {decision_rule.evaluate(correct_vector)}\n")
