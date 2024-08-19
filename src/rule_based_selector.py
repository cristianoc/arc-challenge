from typing import Dict, List, Optional

# Type aliases for clarity
Embedding = Dict[str, int]
Indices = List[str]


class DecisionRule:
    def __init__(self, embedding: Embedding):
        self.embedding = embedding

    def __str__(self) -> str:
        conditions = [
            f"{name} = {self.embedding[name]}" for name in self.embedding.keys()]
        rule = " AND ".join(conditions)
        return f"{rule}."

    def evaluate(self, embedding: Embedding) -> bool:
        """
        Evaluate if the given embedding satisfies the decision rule.
        """
        return all(embedding[i] == self.embedding[i] for i in self.embedding.keys())

    def intersection(self, other: 'DecisionRule') -> Optional['DecisionRule']:
        """
        Find the intersection of two decision rules.
        """
        common_features = set(self.embedding.keys()) & set(
            other.embedding.keys())
        if not common_features:
            return None
        intersection_embedding = {
            i: self.embedding[i] for i in common_features if self.embedding[i] == other.embedding[i]}
        if not intersection_embedding:
            return None
        return DecisionRule(intersection_embedding)


def find_unique_features(embedding: Embedding, other_embeddings: List[Embedding], minimal: bool = False) -> Indices:
    """
    Identify the set of unique features that differentiate the embedding from all the others.
    Returns the indices of the features that are necessary to uniquely identify the correct object.
    If minimal is True, the function will return the minimal set of features that uniquely identify the object.
    """
    all_features = [i for i in embedding if any(
        embedding[i] != other[i] for other in other_embeddings)]
    minimal_features = all_features.copy()

    if minimal:
        for feature in all_features:
            temp_features = [f for f in minimal_features if f != feature]
            temp_correct_vector = [embedding[f] for f in temp_features]
            temp_other_vectors = [[other[f] for f in temp_features]
                                  for other in other_embeddings]
            # Check if the remaining features still uniquely identify the object
            if all(temp_correct_vector != other for other in temp_other_vectors):
                minimal_features.remove(feature)

    return minimal_features


def generate_decision_rule(embedding: Embedding, unique_features: Indices) -> Optional[DecisionRule]:
    """
    Generate the decision rule based on unique features.
    """
    if not unique_features:
        return None
    embedding = {i: embedding[i] for i in unique_features}
    return DecisionRule(embedding)


def select_object_minimal(embeddings: List[Embedding], correct_object_index: int) -> Optional[DecisionRule]:
    """
    Main function to process the experiment and return the minimal selection rule for a given correct object index.
    """
    correct_vector = embeddings[correct_object_index]

    other_vectors = [v for i, v in enumerate(
        embeddings) if i != correct_object_index]

    minimal_unique_features = find_unique_features(
        correct_vector, other_vectors)
    return generate_decision_rule(correct_vector, minimal_unique_features)

# Testing the functions with the scenarios


def test():
    embeddings1: List[Embedding] = [
        {"a": 1, "b": 1, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 0, "b": 1, "c": 1, "d": 0, "e": 1, "f": 0, "g": 0},
        {"a": 1, "b": 0, "c": 0, "d": 1, "e": 1, "f": 1, "g": 0},
    ]

    embeddings2: List[Embedding] = [
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
    ]

    embeddings3: List[Embedding] = [
        {"a": 1, "b": 0, "c": 1, "d": 0, "e": 1, "f": 1, "g": 0},
        {"a": 0, "b": 1, "c": 0, "d": 0, "e": 1, "f": 0, "g": 0},
        {"a": 0, "b": 0, "c": 1, "d": 0, "e": 1, "f": 0, "g": 1},
    ]

    experiments = [
        ("Experiment1", embeddings1, 0),
        ("Experiment1", embeddings1, 2),
        ("Experiment2", embeddings2, 0),
        ("Experiment3", embeddings3, 0),
    ]

    decision_rules = [select_object_minimal(e[1], e[2]) for e in experiments]

    for i, decision_rule in enumerate(decision_rules):
        name, embeddings, index = experiments[i]
        if decision_rule is None:
            print(f"\n{name} #{index}: No unique selection is possible")
        else:
            print(f"\n{name} #{index}: {decision_rule}")
            for i in range(len(embeddings)):
                v = embeddings[i]
                evaluation = decision_rule.evaluate(v)
                print(f"Eval {name} #{i}: {evaluation}")
                assert (i == index) == evaluation, f"Error in {name} #{index}"
