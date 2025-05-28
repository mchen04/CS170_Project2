from utils import euclidean_distance  # use distance function from utils

#basic NN classifier (stores data + picks closest point)
class NearestNeighborClassifier:

    def __init__(self):
        #start with empty training set
        self.training_data = []

    def train(self, training_data):
        #store the training data (assumes it's already normalized)
        self.training_data = training_data

    def test(self, test_instance_feature_vector):
        #predict label for test instance by finding closest neighbor
        if not self.training_data:
            print("Warning: No training data.")
            return None

        min_distance = float('inf')
        predicted_class_label = None

        for train_feature_vector, train_class_label in self.training_data:
            distance = euclidean_distance(test_instance_feature_vector, train_feature_vector)
            if distance < min_distance:
                min_distance = distance
                predicted_class_label = train_class_label

        return predicted_class_label
