from ultralytics import YOLO
from pathlib import Path
import itertools
import json
import csv
import random
import numpy as np
import requests
from typing import Any
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    recall_score,
    f1_score,
    classification_report,
)


def send_update_to_discord(message: str) -> None:
    """Send update message to Discord webhook"""
    url = "https://discord.com/api/webhooks/1443837171323768873/oM4mmdCxFxX09OAL0Z3Akh3BPumVIixnahGMr7_HYD-WsKxmMaruGwQzk8_M6y3BPM76"
    data = {"content": message}
    requests.post(url, json=data)


# Reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

send_update_to_discord("YOLO: init done")


def generate_hyperparameter_combinations(
    hyperparameter_grid: dict[str, list[Any]],
) -> list[dict[str, Any]]:
    """Generate all combinations from hyperparameter grid"""
    keys = hyperparameter_grid.keys()
    values = hyperparameter_grid.values()
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def create_run_name(hyperparams: dict[str, Any]) -> str:
    """Create a run name from hyperparameters"""
    lr_str = str(hyperparams["learning_rate"]).replace(".", "p")
    bs_str = str(hyperparams["batch_size"]).replace(".", "p")
    wd_str = str(hyperparams["weight_decay"]).replace(".", "p")
    do_str = str(hyperparams["dropout_rate"]).replace(".", "p")
    if hyperparams["model_size"] == "yolo11x-cls.pt":
        model_str = "yolo11x"
    else:
        model_str = "yolo11"
    return f"{model_str}_lr{lr_str}_bs{bs_str}_wd{wd_str}_do{do_str}"


def load_image_paths_and_labels_from_test_set(
    dataset_dir: str,
) -> tuple[list[Path], list[int], list[str]]:
    """Load image paths and labels from directory structure"""
    data_dir = Path(dataset_dir) / "test"
    image_paths = []
    labels = []
    class_names = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(class_names)}

    for class_dir in data_dir.iterdir():
        if not class_dir.is_dir():
            continue
        class_idx = class_to_idx[class_dir.name]
        for img_path in class_dir.glob("*"):
            if img_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".gif"]:
                image_paths.append(img_path)
                labels.append(class_idx)

    return image_paths, labels, class_names


def extract_training_history(
    run_name: str,
) -> list[dict[str, Any]]:
    """Extract training and validation history from CSV"""
    train_history = []

    results_csv_path = Path("yolo_runs") / run_name / "results.csv"
    if results_csv_path.exists():
        with open(results_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    train_history.append(
                        {
                            "loss": float(row.get("train/loss", 0)),
                            "accuracy": float(row.get("metrics/accuracy_top1", 0)),
                        }
                    )
                except (ValueError, KeyError):
                    continue

    return train_history


def extract_validation_history(
    run_name: str,
) -> list[dict[str, Any]]:
    """Extract validation history from CSV"""
    validation_history = []
    results_csv_path = Path("yolo_runs") / run_name / "results.csv"
    if results_csv_path.exists():
        with open(results_csv_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    validation_history.append(
                        {
                            "loss": float(row.get("val/loss", 0)),
                        }
                    )
                except (ValueError, KeyError):
                    continue
    return validation_history


def train(
    model: YOLO,
    dataset_dir: str,
    hyperparams: dict[str, Any],
) -> None:
    """Train YOLO model"""
    run_name = create_run_name(hyperparams)
    send_update_to_discord(f"YOLO: Starting training - {run_name}")
    model.train(
        data=dataset_dir,
        epochs=hyperparams["num_epochs"],
        batch=hyperparams["batch_size"],
        optimizer="AdamW",
        lr0=hyperparams["learning_rate"],
        weight_decay=hyperparams["weight_decay"],
        dropout=hyperparams["dropout_rate"],
        patience=hyperparams["patience"],
        imgsz=224,
        hsv_h=0.2,
        hsv_s=0.2,
        hsv_v=0.2,
        degrees=20,
        fliplr=0.5,
        project="yolo_runs",
        name=run_name,
        exist_ok=True,
        device=0,
        seed=SEED,
        verbose=False,
    )


def evaluate(
    model: YOLO, image_paths: list[Path], labels: list[int], hyperparams: dict[str, Any]
) -> tuple[float, float, float, list[int]]:
    """Evaluate YOLO model and return accuracy, recall, f1, predictions"""
    run_name = create_run_name(hyperparams)
    send_update_to_discord(f"YOLO: Starting evaluation - {run_name}")

    image_paths_str = [str(img_path) for img_path in image_paths]
    batch_size = hyperparams["batch_size"]
    all_preds = []
    num_batches = (len(image_paths_str) + batch_size - 1) // batch_size
    for i in tqdm(
        range(0, len(image_paths_str), batch_size),
        desc="Evaluating",
        total=num_batches,
        unit="batch",
        leave=False,
    ):
        batch_paths = image_paths_str[i : i + batch_size]
        batch_results = model.predict(batch_paths, verbose=False, imgsz=224, device=0)
        for result in batch_results:
            if hasattr(result, "probs") and result.probs is not None:
                pred_idx = result.probs.top1
            else:
                pred_idx = 0
            all_preds.append(pred_idx)

    accuracy = accuracy_score(labels, all_preds)
    recall = recall_score(labels, all_preds, average="macro", zero_division=0)
    f1 = f1_score(labels, all_preds, average="macro", zero_division=0)

    return accuracy, recall, f1, all_preds


def is_run_completed(
    hyperparams: dict[str, Any], output_file: str = "yolo_results_v2.json"
) -> bool:
    """Check if a run is already completed by checking if hyperparams exist in JSON results file"""
    if not Path(output_file).exists():
        return False

    try:
        with open(output_file, "r") as f:
            existing_results = json.load(f)

        # Check if hyperparams combination exists in results
        hp_tuple = tuple(sorted(hyperparams.items()))
        for result in existing_results:
            existing_hp = result.get("hyperparameters", {})
            existing_hp_tuple = tuple(sorted(existing_hp.items()))
            if existing_hp_tuple == hp_tuple:
                return True
        return False
    except Exception:
        return False


def train_and_evaluate(
    hyperparams: dict[str, Any],
    dataset_dir: str,
) -> dict[str, Any]:
    """Train model and evaluate on test set"""
    run_name = create_run_name(hyperparams)

    # Load model and train
    model = YOLO(hyperparams["model_size"])
    train(model, dataset_dir, hyperparams)

    # Reload model from checkpoint for evaluation
    # YOLO saves the best model as weights/best.pt in the run directory
    best_model_path = Path("yolo_runs") / run_name / "weights" / "best.pt"
    if best_model_path.exists():
        model = YOLO(str(best_model_path))
    # If best.pt doesn't exist, use the last checkpoint
    else:
        last_model_path = Path("yolo_runs") / run_name / "weights" / "last.pt"
        if last_model_path.exists():
            model = YOLO(str(last_model_path))

    # Evaluate on test set
    test_image_paths, test_labels, class_names = (
        load_image_paths_and_labels_from_test_set(dataset_dir)
    )
    test_acc, test_recall, test_f1, test_preds = evaluate(
        model, test_image_paths, test_labels, hyperparams
    )

    # Generate classification report
    class_report = classification_report(
        test_labels,
        test_preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    send_update_to_discord(
        f"YOLO: Completed - {run_name} | Test Acc: {test_acc:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}"
    )

    return {
        "hyperparameters": hyperparams,
        "test_metrics": {
            "accuracy": test_acc,
            "recall": test_recall,
            "f1_score": test_f1,
        },
        "train_history": extract_training_history(run_name),
        "val_history": extract_validation_history(run_name),
        "classification_report": class_report,
    }


def save_results(
    all_results: list[dict[str, Any]], output_file: str = "yolo_results_v2.json"
) -> None:
    """Save results to JSON file, merging with existing results"""
    # Load existing results if file exists
    existing_results = []
    if Path(output_file).exists():
        try:
            with open(output_file, "r") as f:
                existing_results = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load existing results: {e}")

    # Create a set of existing hyperparameter combinations to avoid duplicates
    existing_hyperparams = set()
    for result in existing_results:
        hp = result.get("hyperparameters", {})
        hp_tuple = tuple(sorted(hp.items()))
        existing_hyperparams.add(hp_tuple)

    # Merge new results with existing ones (avoid duplicates)
    merged_results = existing_results.copy()
    for result in all_results:
        hp = result.get("hyperparameters", {})
        hp_tuple = tuple(sorted(hp.items()))
        if hp_tuple not in existing_hyperparams:
            merged_results.append(result)
            existing_hyperparams.add(hp_tuple)
        else:
            # Update existing result with new one (in case it was re-run)
            for i, existing_result in enumerate(merged_results):
                existing_hp = existing_result.get("hyperparameters", {})
                existing_hp_tuple = tuple(sorted(existing_hp.items()))
                if existing_hp_tuple == hp_tuple:
                    merged_results[i] = result
                    break

    # Save merged results
    with open(output_file, "w") as f:
        json.dump(merged_results, f, indent=2)


def print_summary(all_results: list[dict[str, Any]]) -> None:
    """Print summary of all results"""
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for i, result in enumerate(all_results, 1):
        hp = result["hyperparameters"]
        metrics = result["test_metrics"]
        print(f"\nCombination {i}:")
        print(
            f"Model: {hp['model_size']}, LR: {hp['learning_rate']}, BS: {hp['batch_size']}, WD: {hp['weight_decay']}, Dropout: {hp['dropout_rate']}"
        )
        print(
            f"Test Accuracy: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f}, F1: {metrics['f1_score']:.4f}"
        )

    # Find best combination
    if not all_results:
        print("\nWARNING: No successful training runs completed!")
        print("Check error messages above for details.")
        return

    best_result = max(all_results, key=lambda x: x["test_metrics"]["f1_score"])
    metrics = best_result["test_metrics"]
    hp = best_result["hyperparameters"]
    print(f"\nBest combination (F1: {metrics['f1_score']:.4f}):")
    print(f"Hyperparameters: {best_result['hyperparameters']}")
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Test Recall: {metrics['recall']:.4f}")

    send_update_to_discord(
        f"YOLO: BEST RESULT - F1: {metrics['f1_score']:.4f}, Acc: {metrics['accuracy']:.4f}, Recall: {metrics['recall']:.4f} | HP: {hp}"
    )
    send_update_to_discord("YOLO: All training completed!")


def main():
    """Main function"""
    import argparse

    parser = argparse.ArgumentParser(description="Train YOLO classification model")
    parser.add_argument(
        "--job_id",
        type=int,
        default=0,
        help="Job ID for parallel execution (0-based, default: 0)",
    )
    parser.add_argument(
        "--total_jobs",
        type=int,
        default=1,
        help="Total number of parallel jobs (default: 1)",
    )

    args = parser.parse_args()

    dataset_dir = "temp_yolo_dataset"

    # Hyperparameter grid
    hyperparameter_grid = {
        "model_size": ["yolo11l-cls.pt", "yolo11x-cls.pt"],
        "learning_rate": [0.00005, 0.0001, 0.0002],
        "batch_size": [32, 64],
        "weight_decay": [0.0, 0.0001, 0.001],
        "dropout_rate": [0.1, 0.2, 0.3],
        "num_epochs": [50],
        "patience": [5],
    }

    # Generate all combinations
    combinations = generate_hyperparameter_combinations(hyperparameter_grid)
    # For testing only
    # combinations = combinations[:1]
    # combinations[0]["num_epochs"] = 1

    # Filter out combinations that already have results
    output_file = "yolo_results_v2.json"
    remaining_combinations = []
    for combo in combinations:
        if not is_run_completed(combo, output_file):
            remaining_combinations.append(combo)
        else:
            send_update_to_discord(
                f"YOLO: skipping combination {create_run_name(combo)}"
            )
            print(f"YOLO: skipping combination {create_run_name(combo)}")

    print(f"Total hyperparameter combinations: {len(combinations)}")
    print(f"Already completed: {len(combinations) - len(remaining_combinations)}")
    print(f"Remaining: {len(remaining_combinations)}")

    # Split remaining combinations for parallel execution
    if args.total_jobs > 1:
        chunk_size = len(remaining_combinations) // args.total_jobs
        start_idx = args.job_id * chunk_size
        if args.job_id == args.total_jobs - 1:
            # Last job takes remaining combinations
            end_idx = len(remaining_combinations)
        else:
            end_idx = start_idx + chunk_size
        remaining_combinations = remaining_combinations[start_idx:end_idx]
        print(
            f"Job {args.job_id}/{args.total_jobs}: Processing {len(remaining_combinations)} combinations (indices {start_idx}-{end_idx - 1})"
        )
        send_update_to_discord(
            f"YOLO: Job {args.job_id}/{args.total_jobs}: Processing {len(remaining_combinations)} combinations"
        )
    else:
        send_update_to_discord(
            f"YOLO: Starting hyperparameter search with {len(remaining_combinations)} combinations"
        )

    all_results = []

    # Train and evaluate for each remaining combination
    for i, hyperparams in enumerate(remaining_combinations, 1):
        print(f"\n{'=' * 60}")
        print(f"Combination {i}/{len(remaining_combinations)}")
        print(f"Hyperparameters: {hyperparams}")
        print(f"{'=' * 60}")
        send_update_to_discord(
            f"YOLO: Combination {i}/{len(remaining_combinations)} starting"
        )

        try:
            result = train_and_evaluate(hyperparams, dataset_dir)
            all_results.append(result)

            metrics = result["test_metrics"]
            print("\nTest Results:")
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Recall: {metrics['recall']:.4f}")
            print(f"F1 Score: {metrics['f1_score']:.4f}")

            save_results(all_results)
        except Exception as e:
            print(f"Error with combination {i}: {e}")
            send_update_to_discord(f"YOLO: ERROR in combination {i}: {str(e)}")
            import traceback

            traceback.print_exc()
            continue

    print_summary(all_results)


if __name__ == "__main__":
    main()
