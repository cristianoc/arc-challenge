from typing import Dict, List, Optional
from grid_data import logger

# Type aliases for clarity

Features = Dict[str, int | bool]
Indices = List[str]


class DecisionRule:
    def __init__(self, features: Features):
        self.features = {key: features[key] for key in sorted(features.keys())}

    def __str__(self) -> str:
        conditions = [
            f"{name} = {self.features[name]}" for name in self.features.keys()
        ]
        rule = " AND ".join(conditions)
        return f"{rule}."

    def evaluate(self, features: Features) -> bool:
        """
        Evaluate if the given features satisfy the decision rule.
        """
        return all(features[i] == self.features[i] for i in self.features.keys())

    def intersection(self, other: "DecisionRule") -> Optional["DecisionRule"]:
        """
        Find the intersection of two decision rules.
        """
        common_features = set(self.features.keys()) & set(other.features.keys())
        if not common_features:
            return None
        intersection_features = {
            i: self.features[i]
            for i in common_features
            if self.features[i] == other.features[i]
        }
        if not intersection_features:
            return None
        return DecisionRule(intersection_features)


def find_unique_features(
    features: Features, other_features: List[Features], minimal: bool = False
) -> Indices:
    """
    Identify the set of unique features that differentiate the features from all the others.
    Returns the indices of the features that are necessary to uniquely identify the correct object.
    If minimal is True, the function will return the minimal set of features that uniquely identify the object.
    """
    all_features = [
        i for i in features if any(features[i] != other[i] for other in other_features)
    ]
    minimal_features = all_features.copy()

    if minimal:
        for feature in all_features:
            temp_features = [f for f in minimal_features if f != feature]
            temp_correct_vector = [features[f] for f in temp_features]
            temp_other_vectors = [
                [other[f] for f in temp_features] for other in other_features
            ]
            # Check if the remaining features still uniquely identify the object
            if all(temp_correct_vector != other for other in temp_other_vectors):
                minimal_features.remove(feature)

    return minimal_features


def generate_decision_rule(
    features: Features, unique_features: Indices
) -> Optional[DecisionRule]:
    """
    Generate the decision rule based on unique features.
    """
    if not unique_features:
        return None
    features = {i: features[i] for i in unique_features}
    return DecisionRule(features)


def select_object_minimal(
    features_list: List[Features], correct_object_index: int
) -> Optional[DecisionRule]:
    """
    Main function to process the experiment and return the minimal selection rule for a given correct object index.
    """
    correct_vector = features_list[correct_object_index]

    other_vectors = [
        v for i, v in enumerate(features_list) if i != correct_object_index
    ]

    minimal_unique_features = find_unique_features(correct_vector, other_vectors)
    return generate_decision_rule(correct_vector, minimal_unique_features)


# Testing the functions with the scenarios


def test():
    features1: List[Features] = [
        {"a": 1, "b": 1, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 0, "b": 1, "c": 1, "d": 0, "e": 1, "f": 0, "g": 0},
        {"a": 1, "b": 0, "c": 0, "d": 1, "e": 1, "f": 1, "g": 0},
    ]

    features2: List[Features] = [
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
    ]

    features3: List[Features] = [
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 0, "b": 1, "c": 0, "d": 0, "e": 1, "f": 0, "g": 0},
        {"a": 0, "b": 0, "c": 1, "d": 0, "e": 1, "f": 0, "g": 1},
    ]

    experiments = [
        ("Experiment1", features1, 0),
        ("Experiment1", features1, 2),
        ("Experiment2", features2, 0),
        ("Experiment3", features3, 0),
    ]

    decision_rules = [select_object_minimal(e[1], e[2]) for e in experiments]

    for i, decision_rule in enumerate(decision_rules):
        name, features_list, index = experiments[i]
        if decision_rule is None:
            logger.info(f"\n{name} #{index}: No unique selection is possible")
        else:
            logger.info(f"\n{name} #{index}: {decision_rule}")
            for i in range(len(features_list)):
                v = features_list[i]
                evaluation = decision_rule.evaluate(v)
                logger.info(f"Eval {name} #{i}: {evaluation}")
                assert (i == index) == evaluation, f"Error in {name} #{index}"
