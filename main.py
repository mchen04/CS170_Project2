import random

#rand function
def test_evaluate_subset(x):
    return random.uniform(0.0, 100.0)

#forward selection function
def greedy_forward_selection(num_features, evaluate_func):
    current_feature_set = set()  
    trace_log = []  #store log

    #evaluate
    best_overall_accuracy = evaluate_func(current_feature_set)
    best_overall_feature_set = set(current_feature_set)
    trace_log.append(f"Using no features and random evaluation, I get an accuracy of {best_overall_accuracy:.1f}%\nBeginning search.")

    for _ in range(num_features):
        feature_candidates = [f for f in range(1, num_features + 1) if f not in current_feature_set]
        best_accuracy = -1
        best_feature = None
        temp_log = []

        for feature in feature_candidates:
            candidate_feature_set = current_feature_set | {feature}
            accuracy = evaluate_func(candidate_feature_set)
            temp_log.append(f"Using feature(s) {{{', '.join(map(str, sorted(current_feature_set)))}}} accuracy is {accuracy:.1f}%")
            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_feature = feature

        trace_log.extend(temp_log)

        if best_feature is not None:
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
    current_feature_set = set(range(1, num_features + 1))  # Start with all features
    trace_log = []

    # Evaluate the full set
    best_overall_accuracy = evaluate_func(current_feature_set)
    best_overall_feature_set = set(current_feature_set)
    trace_log.append(f"Using all features and random evaluation, I get an accuracy of {best_overall_accuracy:.1f}%\nBeginning search.")

    for _ in range(num_features - 1):
        feature_candidates = list(current_feature_set)
        best_accuracy = float('inf') 
        feature_to_remove = None
        temp_log = []

        for feature in feature_candidates:
            candidate_feature_set = current_feature_set - {feature}
            accuracy = evaluate_func(candidate_feature_set)

            temp_log.append(f"Using feature(s) {{{', '.join(map(str, sorted(candidate_feature_set)))}}} accuracy is {accuracy:.1f}%")

            if accuracy < best_accuracy:
                best_accuracy = accuracy
                feature_to_remove = feature

        trace_log.extend(temp_log)

        if feature_to_remove is not None:
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
    print("Welcome to Tabito Sakamoto's Feature Selection Algorithm.")

    try:
        num_features = input("Please enter total number of features: ")
    except ValueError:
        print("Invalid input. Please enter an integer.")
        return

    print("Type the number of the algorithm you want to run.")
    print("1) Forward Selection")
    print("2) Backward Elimination")

    choice = input()

    if choice == "1":
        print("Forward Selection will run here.")  
    elif choice == "2":
        print("Backward Elimination will run here.")  
    else:
        print("Invalid input. Please enter 1 or 2.")