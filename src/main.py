import random
from data_loader import load_dataset, normalize_features
from validator import evaluate_loocv
from nn_classifier import NearestNeighborClassifier

#rand function
def test_evaluate_subset(feature_subset):
    #using dummy evaluation 
    _ = feature_subset
    return random.uniform(0.0, 100.0)

#wrapper function for evaluate_loocv with k parameter
def create_real_evaluate_func(dataset, k=1):
    def real_evaluate_func(feature_subset):
        if len(feature_subset) == 0:
            #return a low accuracy for empty feature set
            return random.uniform(0.0, 10.0)
        #convert set to list for indexing
        feature_list = list(feature_subset)
        #search uses 1-based indexing + evaluate_loocv handles conversion to 0-based.
        accuracy = evaluate_loocv(dataset, feature_list, lambda: NearestNeighborClassifier(k))
        return accuracy
    return real_evaluate_func

#forward selection function
def greedy_forward_selection(num_features, evaluate_func):
    current_feature_set = set()  
    trace_log = []  #store log

    #evaluate
    best_overall_accuracy = evaluate_func(current_feature_set)
    best_overall_feature_set = set(current_feature_set)
    trace_log.append(f"Running nearest neighbor with no features (default rate), using \"leaving-one-out\" evaluation, I get an accuracy of {best_overall_accuracy:.1f}%\nBeginning search.")

    for _ in range(num_features):
        feature_candidates = [f for f in range(1, num_features + 1) if f not in current_feature_set]
        best_accuracy = -1
        best_features = []
        temp_log = []

        for feature in feature_candidates:
            candidate_feature_set = current_feature_set | {feature}
            accuracy = evaluate_func(candidate_feature_set)
            temp_log.append(f"Using feature(s) {{{', '.join(map(str, sorted(list(candidate_feature_set))))}}} accuracy is {accuracy:.1f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features = [feature]
            elif accuracy == best_accuracy:
                best_features.append(feature)

        trace_log.extend(temp_log)

        if best_features:
            best_feature = max(best_features)  
            current_feature_set.add(best_feature)  

            trace_log.append(f"Feature set {{{', '.join(map(str, sorted(current_feature_set)))}}} was best, accuracy is {best_accuracy:.1f}%")

        if best_accuracy < best_overall_accuracy:
            trace_log.append("(Warning: Decreased accuracy! )")
        else:
            best_overall_accuracy = best_accuracy
            best_overall_feature_set = set(current_feature_set)

    trace_log.append(f"Search finished! The best subset of features is {{{', '.join(map(str, sorted(best_overall_feature_set)))}}}, which has an accuracy of {best_overall_accuracy:.1f}%")
    return trace_log

#backward function
def backward_elimination(num_features, evaluate_func):
    current_feature_set = set(range(1, num_features + 1))  # start with all features
    trace_log = []

    # evaluate the full set
    best_overall_accuracy = evaluate_func(current_feature_set)
    best_overall_feature_set = set(current_feature_set)
    trace_log.append(f"Running nearest neighbor with all features, using \"leaving-one-out\" evaluation, I get an accuracy of {best_overall_accuracy:.1f}%\nBeginning search.")

    for _ in range(num_features - 1):
        feature_candidates = list(current_feature_set)
        best_accuracy = -1
        best_features_to_remove = []
        temp_log = []

        for feature in feature_candidates:
            candidate_feature_set = current_feature_set - {feature}
            accuracy = evaluate_func(candidate_feature_set)

            temp_log.append(f"Using feature(s) {{{', '.join(map(str, sorted(candidate_feature_set)))}}} accuracy is {accuracy:.1f}%")

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_features_to_remove = [feature]
            elif accuracy == best_accuracy:
                best_features_to_remove.append(feature)

        trace_log.extend(temp_log)

        if best_features_to_remove:
            feature_to_remove = min(best_features_to_remove)  
            current_feature_set.remove(feature_to_remove)
            trace_log.append(f"Feature set {{{', '.join(map(str, sorted(current_feature_set)))}}} was best, accuracy is {best_accuracy:.1f}%")

        if best_accuracy < best_overall_accuracy:
            trace_log.append("(Warning: Decreased accuracy! )")
        else:
            best_overall_accuracy = best_accuracy
            best_overall_feature_set = set(current_feature_set)

    trace_log.append(f"Search finished! The best subset of features is {{{', '.join(map(str, sorted(best_overall_feature_set)))}}}, which has an accuracy of {best_overall_accuracy:.1f}%")
    return trace_log

#main ui
def main():
    print("Welcome to Tabito Sakamoto & Michael Chen's Feature Selection Algorithm.")
    
    file_path = input("Type in the name of the file to test: ")
    
    #try to load the dataset
    try:
        raw_data = load_dataset(f"data/{file_path}")
        num_features = len(raw_data[0][0])  
        print(f"This dataset has {num_features} features (not including the class attribute), with {len(raw_data)} instances.")
        
        #normalize the features
        print("Please wait while I normalize the data... Done!")
        normalized_data = normalize_features(raw_data)
        
    except FileNotFoundError:
        print(f"Error: Could not find file 'data/{file_path}'. Please make sure the file exists.")
        return
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return

    print("\nType the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")
    print("3) KNN Analysis (k=1,3,5,7)")

    choice = input()

    if choice == "1":
        #create the real evaluation function with k=1
        real_evaluate_func = create_real_evaluate_func(normalized_data, k=1)
        trace = greedy_forward_selection(num_features, real_evaluate_func)
        print("\n".join(trace))
    elif choice == "2":
        #create the real evaluation function with k=1
        real_evaluate_func = create_real_evaluate_func(normalized_data, k=1)
        trace = backward_elimination(num_features, real_evaluate_func)  
        print("\n".join(trace))
    elif choice == "3":
        print("\n" + "="*60)
        print("KNN COMPARISON (Both Algorithms)")
        print("="*60)
        
        k_values = [1, 3, 5, 7]
        forward_results = {}
        backward_results = {}
        
        for k in k_values:
            print(f"\n--- Testing with k={k} ---")
            real_evaluate_func = create_real_evaluate_func(normalized_data, k=k)
            
            print(f"Forward Selection (k={k}):")
            trace = greedy_forward_selection(num_features, real_evaluate_func)
            forward_result = trace[-1]  #get final result line
            print(forward_result)
            forward_results[k] = forward_result
            
            print(f"\nBackward Elimination (k={k}):")
            trace = backward_elimination(num_features, real_evaluate_func)
            backward_result = trace[-1]  #get final result line
            print(backward_result)
            backward_results[k] = backward_result
        
        #summary comparison
        print("\n" + "="*60)
        print("SUMMARY COMPARISON")
        print("="*60)
        print("k\tForward Selection Result")
        print("-" * 60)
        
        for k in k_values:
            print(f"{k}\t{forward_results[k]}")
            
        print("\nk\tBackward Elimination Result")
        print("-" * 60)
        
        for k in k_values:
            print(f"{k}\t{backward_results[k]}")
    else:
        print("Invalid input. Please enter 1, 2, or 3.")


if __name__ == "__main__":
    main()