import logging
from typing import Any, Dict

from matplotlib import pyplot as plt
from grid_data import logger
from load_data import training_data, evaluation_data
from predict_size import Config, compute_perc_correct, num_difficulties_total, process_tasks

from datetime import datetime
import concurrent.futures
import json

plt: Any = plt

def evaluate_difficulty(difficulty_level: int):
    """Evaluate the model on the training and evaluation data for a given difficulty level."""
    Config.difficulty = difficulty_level
    logger.setLevel("ERROR")
    num_correct_tr, num_incorrect_tr = process_tasks(
        training_data, "training_data")
    num_correct_ev, num_incorrect_ev = process_tasks(
        evaluation_data, "evaluation_data")

    perc_correct_tr = compute_perc_correct(num_correct_tr, num_incorrect_tr)
    perc_correct_ev = compute_perc_correct(num_correct_ev, num_incorrect_ev)

    # Return results for this difficulty level
    return difficulty_level, perc_correct_tr, perc_correct_ev


def get_current_time() -> str:
    return datetime.now().strftime('%H:%M:%S.%f')[:-3]


def ablation_study():

    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        logger.addHandler(console_handler)

    # Set a formatter that includes the timestamp
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    results: Dict[int, Any] = {}

    # Run evaluations in parallel for each difficulty level
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_difficulty, difficulty_level)
                   for difficulty_level in range(1, num_difficulties_total+1)]
        for future in concurrent.futures.as_completed(futures):
            difficulty_level, perc_correct_tr, perc_correct_ev = future.result()
            results[difficulty_level] = {
                "training_data": perc_correct_tr,
                "evaluation_data": perc_correct_ev
            }

    # Sort results by difficulty level
    # Sorting ensures the difficulty levels are in order after parallel processing.
    results = dict(sorted(results.items()))
      

    # Write summary of results to JSON file with sorted keys
    with open("ablation_results.json", "w") as f:
        json.dump(results, f, indent=4)

    # Extracting levels and corresponding accuracies
    levels = list(results.keys())  # Ensure levels are sorted
    training_accuracies = [results[level]["training_data"] for level in levels]
    evaluation_accuracies = [results[level]
                             ["evaluation_data"] for level in levels]

    # Plotting the results
    plt.figure(figsize=(10, 6))
    plt.plot(levels, training_accuracies,
             label="Training Data Accuracy", marker='o')
    plt.plot(levels, evaluation_accuracies,
             label="Evaluation Data Accuracy", marker='o')
    plt.xlabel("Model's Difficulty Level")
    plt.ylabel("Accuracy (%)")
    plt.title("Ablation Study: Accuracy vs. Model's Difficulty Level")
    plt.legend()
    plt.grid(True)
    plt.xticks(levels)
    plt.ylim(60, 100)
    plt.show()

    logger.error(
        f"{get_current_time()} - Ablation study completed. Results saved to 'ablation_results.json'.")


if __name__ == "__main__":
    ablation_study()
