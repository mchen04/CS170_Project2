import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_dataset, normalize_features
from validator import evaluate_loocv
from nn_classifier import NearestNeighborClassifier
from main import greedy_forward_selection, backward_elimination

#wrapper function for evaluate_loocv with k parameter
def create_knn_evaluate_func(dataset, k=1):
    def knn_evaluate_func(feature_subset):
        if len(feature_subset) == 0:
            #return a low accuracy for empty feature set
            return 5.0
        #convert set to list for indexing
        feature_list = list(feature_subset)
        #search uses 1-based indexing + evaluate_loocv handles conversion to 0-based
        accuracy = evaluate_loocv(dataset, feature_list, lambda: NearestNeighborClassifier(k))
        return accuracy
    return knn_evaluate_func

#test KNN with different k values on Titanic dataset
def test_knn_titanic():
    print("Loading Titanic dataset...")
    
    #load and normalize data
    raw_data = load_dataset("data/titanic clean.txt")
    num_features = len(raw_data[0][0])
    print(f"Dataset has {num_features} features, {len(raw_data)} instances")
    
    normalized_data = normalize_features(raw_data)
    print("Data normalized.")
    
    print("\n" + "="*60)
    print("KNN COMPARISON ON TITANIC DATASET (Forward Selection Only)")
    print("="*60)
    
    k_values = [1, 3, 5, 7]
    results = {}
    
    for k in k_values:
        print(f"\n--- Testing with k={k} ---")
        knn_evaluate_func = create_knn_evaluate_func(normalized_data, k)
        
        #test forward selection only (faster)
        print(f"Forward Selection (k={k}):")
        trace = greedy_forward_selection(num_features, knn_evaluate_func)
        forward_result = trace[-1]  #get final result line
        print(forward_result)
        
        #store results for comparison
        results[k] = forward_result
    
    #summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON - Forward Selection")
    print("="*60)
    print("k\tResult")
    print("-" * 40)
    
    for k in k_values:
        print(f"{k}\t{results[k]}")

if __name__ == "__main__":
    test_knn_titanic()