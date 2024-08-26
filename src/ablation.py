import logging
from grid_data import logger
from load_data import training_data, evaluation_data
from predict_size import Config, compute_perc_correct, process_tasks

from datetime import datetime

def ablation_study():
    logger.setLevel("ERROR")

    if not logger.hasHandlers():
            console_handler = logging.StreamHandler()
            logger.addHandler(console_handler)

    # Set a formatter that includes the timestamp
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    for handler in logger.handlers:
        handler.setFormatter(formatter)

    results = {}

    def get_current_time() -> str:
        return datetime.now().strftime('%H:%M:%S.%f')[:-3] # type: ignore

    # Iterate over difficulty levels from 1 to 12
    for difficulty_level in range(1, 13):
        current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3] # type: ignore
        logger.error(f"{get_current_time()} - Evaluating difficulty level: {difficulty_level}")

        Config.difficulty = difficulty_level
        num_correct_tr, num_incorrect_tr = process_tasks(training_data, "training_data")
        num_correct_ev, num_incorrect_ev = process_tasks(evaluation_data, "evaluation_data")
        perc_correct_tr = compute_perc_correct(num_correct_tr, num_incorrect_tr)
        perc_correct_ev = compute_perc_correct(num_correct_ev, num_incorrect_ev)

        # Store results for this difficulty level
        results[difficulty_level] = {
            "training_data": perc_correct_tr,
            "evaluation_data": perc_correct_ev
        }
    
    # Write summary of results to JSON file
    with open("ablation_results.json", "w") as f:
        import json
        json.dump(results, f, indent=4)

    current_time = datetime.now().strftime('%H:%M:%S.%f')[:-3] # type: ignore
    logger.error(f"\n{get_current_time()} - Ablation study completed. Results saved to 'ablation_results.json'.")

if __name__ == "__main__":
    ablation_study()