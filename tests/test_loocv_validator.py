import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.data_loader import load_dataset, normalize_features
from src.nn_classifier import NearestNeighborClassifier
from src.validator import evaluate_loocv 

def run_loocv_tests():
    print("\nTesting Leave-One-Out Cross-Validation (LOOCV) Validator")

    # get paths
    current_dir = os.path.dirname(__file__)
    small_data_filepath = os.path.join(current_dir, '..', 'data', 'small-test-dataset.txt')
    large_data_filepath = os.path.join(current_dir, '..', 'data', 'large-test-dataset.txt')

    # small dataset
    print(f"\nLoading and normalizing small dataset from: {small_data_filepath}")
    raw_small_dataset = load_dataset(small_data_filepath)
    normalized_small_dataset = normalize_features(raw_small_dataset)

    if not normalized_small_dataset:
        print("Couldn't load or normalize small dataset.")
    else:
        print("\nRunning LOOCV on small dataset with features {3, 5, 7}...")
        feature_subset_small = [3, 5, 7]
        accuracy_small = evaluate_loocv(normalized_small_dataset, feature_subset_small, NearestNeighborClassifier)
        print(f"Accuracy: {accuracy_small:.3f}%")

        expected_accuracy_small = 89.0
        if abs(accuracy_small - expected_accuracy_small) < 1.0:
            print("Accuracy looks about right")
        else:
            print(f"Unexpected accuracy (expected ~{expected_accuracy_small:.1f}%)")

    # large dataset
    print(f"\nLoading and normalizing large dataset from: {large_data_filepath}")
    raw_large_dataset = load_dataset(large_data_filepath)
    normalized_large_dataset = normalize_features(raw_large_dataset)

    if not normalized_large_dataset:
        print("Couldn't load or normalize large dataset.")
    else:
        print("\nRunning LOOCV on large dataset with features {1, 15, 27}...")
        feature_subset_large = [1, 15, 27]
        accuracy_large = evaluate_loocv(normalized_large_dataset, feature_subset_large, NearestNeighborClassifier)
        print(f"Accuracy: {accuracy_large:.3f}%")

        expected_accuracy_large = 94.9
        if abs(accuracy_large - expected_accuracy_large) < 1.0:
            print("Accuracy looks about right")
        else:
            print(f"Unexpected accuracy (expected ~{expected_accuracy_large:.1f}%)")

    print("\n--- LOOCV Validator Tests Complete ---")

if __name__ == "__main__":
    run_loocv_tests()