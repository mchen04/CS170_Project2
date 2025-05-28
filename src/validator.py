#evaluates accuracy using LOOCV for the given feature subset
def evaluate_loocv(dataset, feature_subset, classifier_class):

    if not dataset:
        return 0.0  #no data, no accuracy

    total_instances = len(dataset)
    correct_predictions = 0

    #shift features to 0-based index
    adjusted_feature_subset = [f - 1 for f in feature_subset]

    for i in range(total_instances):
        #pick one instance to leave out
        test_instance_full_vector, actual_label = dataset[i]

        #use rest as training data
        training_set = dataset[:i] + dataset[i+1:]

        #pick only selected features from test instance
        test_instance_features = [test_instance_full_vector[j] for j in adjusted_feature_subset]

        #pick only selected features from training set
        training_data = []
        for train_vector, label in training_set:
            selected_features = [train_vector[j] for j in adjusted_feature_subset]
            training_data.append((selected_features, label))

        #create and train model for this fold
        classifier = classifier_class()
        classifier.train(training_data)

        #predict
        predicted_label = classifier.test(test_instance_features)

        # heck prediction
        if predicted_label == actual_label:
            correct_predictions += 1

    #return accuracy %
    accuracy = (correct_predictions / total_instances) * 100
    return accuracy
