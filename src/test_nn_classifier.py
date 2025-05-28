import os
from data_loader import load_dataset, normalize_features
from nn_classifier import NearestNeighborClassifier

def run_nn_classifier_tests():
    print("\nTesting Nearest Neighbor Classifier")
    
    # get path to dataset
    current_dir = os.path.dirname(__file__)
    data_filepath = os.path.join(current_dir, '..', 'data', 'small-test-dataset.txt')

    # load + normalize
    print(f"\nLoading and normalizing dataset from: {data_filepath}")
    raw_dataset = load_dataset(data_filepath)
    normalized_dataset = normalize_features(raw_dataset)
    
    if not normalized_dataset:
        print("No data found or failed to normalize. Can't run NN test.")
        return

    # create classifier and train
    print("\nCreating classifier and training on subset")
    classifier = NearestNeighborClassifier()

    training_subset_size = 5
    if len(normalized_dataset) < training_subset_size + 1:
        print(f"Not enough data ({len(normalized_dataset)} instances). Need at least 6.")
        return

    training_data = normalized_dataset[:training_subset_size]
    classifier.train(training_data)

    print(f"Trained on {len(training_data)} instances.")
    print("First 2 training examples:")
    for i in range(min(2, len(training_data))):
        print(f"Features: {training_data[i][0]}, Class: {training_data[i][1]}")

    # run test on 6th instance
    print("\nTesting classifier on 6th instance")
    test_instance_index = training_subset_size
    test_instance_features = normalized_dataset[test_instance_index][0]
    actual_label = normalized_dataset[test_instance_index][1]

    predicted_label = classifier.test(test_instance_features)

    print(f"Test instance features: {test_instance_features}")
    print(f"Actual label: {actual_label}")
    print(f"Predicted label: {predicted_label}")

    #check result
    if predicted_label is not None:
        if predicted_label == actual_label:
            print("Prediction: CORRECT!")
        else:
            print("Prediction: WRONG.")
    else:
        print("Prediction: None made.")

    print("\nNearest Neighbor Classifier Tests Complete")

if __name__ == "__main__":
    run_nn_classifier_tests()
