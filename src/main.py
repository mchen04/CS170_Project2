import random
from data_loader import load_dataset, normalize_features
from validator import evaluate_loocv
from nn_classifier import NearestNeighborClassifier

#rand function
def test_evaluate_subset(feature_subset):
    #using dummy evaluation 
    _ = feature_subset
    return random.uniform(0.0, 100.0)

#wrapper function for evaluate_loocv 
def create_real_evaluate_func(dataset):
    def real_evaluate_func(feature_subset):
        if len(feature_subset) == 0:
            #return a low accuracy for empty feature set
            return random.uniform(0.0, 10.0)
        #convert set to list for indexing
        feature_list = list(feature_subset)
        #search uses 1-based indexing + evaluate_loocv handles conversion to 0-based.
        accuracy = evaluate_loocv(dataset, feature_list, NearestNeighborClassifier)
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
    
    #ask user if they want to use a dataset or just test with dummy evaluation
    print("\nWould you like to:")
    print("1) Test with dummy evaluation (Part I demo)")
    print("2) Use real dataset with actual evaluation (Part II integration)")
    
    mode = input("\nPlease enter your choice (1 or 2): ")
    
    if mode == "1":
        # Part I mode 
        try:
            num_features = int(input("Please enter total number of features: "))
        except ValueError:
            print("Invalid input. Please enter an integer.")
            return
        
        print("\nType the number of the algorithm you want to run.")
        print("1) Forward Selection")
        print("2) Backward Elimination")
        
        choice = input()
        
        if choice == "1":
            trace = greedy_forward_selection(num_features, test_evaluate_subset)
            print("\n".join(trace))
        elif choice == "2":
            trace = backward_elimination(num_features, test_evaluate_subset)
            print("\n".join(trace))
        else:
            print("Invalid input. Please enter 1 or 2.")
            
    elif mode == "2":
        # Part II mode 
        file_path = input("Please enter the dataset file name (e.g., small-test-dataset.txt): ")
        
        #try to load the dataset
        try:
            raw_data = load_dataset(f"data/{file_path}")
            num_features = len(raw_data[0][0])  
            print(f"This dataset has {num_features} features (not including the class attribute), with {len(raw_data)} instances.")
            
            #normalize the features
            print("Please wait while I normalize the data... Done!")
            normalized_data = normalize_features(raw_data)
            
            #create the real evaluation function with the normalized dataset
            real_evaluate_func = create_real_evaluate_func(normalized_data)
            
        except FileNotFoundError:
            print(f"Error: Could not find file 'data/{file_path}'. Please make sure the file exists.")
            return
        except Exception as e:
            print(f"Error loading dataset: {e}")
            return

        print("\nType the number of the algorithm you want to run.")
        print("1) Forward Selection")
        print("2) Backward Elimination")

        choice = input()

        if choice == "1":
            print("\nRunning forward selection...")
            trace = greedy_forward_selection(num_features, real_evaluate_func)
            print("\n".join(trace))
        elif choice == "2":
            print("\nRunning backward elimination...")
            trace = backward_elimination(num_features, real_evaluate_func)  
            print("\n".join(trace))
        else:
            print("Invalid input. Please enter 1 or 2.")
    else:
        print("Invalid choice. Please enter 1 or 2.")


if __name__ == "__main__":
    main()