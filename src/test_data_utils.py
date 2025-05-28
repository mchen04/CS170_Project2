import os
from data_loader import load_dataset, normalize_features
from utils import euclidean_distance

def run_data_utils_tests():
    print("\nTesting Data Loading, Normalization, and Distance Utilities")
    
    #get path to dataset
    current_dir = os.path.dirname(__file__)
    data_filepath = os.path.join(current_dir, '..', 'data', 'small-test-dataset.txt')

    #test loading
    print(f"\nLoading dataset from: {data_filepath}")
    raw_dataset = load_dataset(data_filepath)
    print(f"Loaded {len(raw_dataset)} instances.")
    
    if raw_dataset:
        print("First raw 3 instances:")
        for i in range(min(3, len(raw_dataset))):
            print(f"  Features: {raw_dataset[i][0]}, Class: {raw_dataset[i][1]}")
    else:
        print("No data loaded. Make sure the file exists.")
        return #stop if no data

    #test normalization
    print("\nNormalizing features")
    normalized_dataset = normalize_features(raw_dataset)

    if normalized_dataset:
        print("First 3 normalized instances:")
        for i in range(min(3, len(normalized_dataset))):
            print(f"Features: {normalized_dataset[i][0]}, Class: {normalized_dataset[i][1]}")

        #check if values are within [0, 1]
        print("\nChecking first feature value range (should be in [0,1]):")
        for i in range(min(3, len(normalized_dataset))):
            if normalized_dataset[i][0]:  #make sure not empty
                val = normalized_dataset[i][0][0]
                print(f"  Instance {i} first feature: {val}")
                if not (0.0 <= val <= 1.0):
                    print(f"WARNING: Value {val} is out of range!")
    else:
        print("No data to normalize.")

    #test distance calculation
    print("\nTesting Euclidean distance:")
    if len(normalized_dataset) >= 2:
        point_a = normalized_dataset[0][0]
        point_b = normalized_dataset[1][0]
        dist = euclidean_distance(point_a, point_b)
        print(f"Point A: {point_a}")
        print(f"Point B: {point_b}")
        print(f"Distance: {dist}")
        if dist < 0:
            print("ERROR: Distance should never be negative.")
    elif len(normalized_dataset) == 1:
        print("Only one instance available. Need 2 for distance test.")
    else:
        print("No normalized data to test distance.")

    print("\nData Utils Tests Complete")

if __name__ == "__main__":
    run_data_utils_tests()
