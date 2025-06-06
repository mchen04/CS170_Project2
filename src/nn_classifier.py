from utils import euclidean_distance  # use distance function from utils
from collections import Counter

#basic NN/KNN classifier (stores data + picks k closest points)
class NearestNeighborClassifier:

    def __init__(self, k=1):
        #start with empty training set
        self.training_data = []
        self.k = k

    def train(self, training_data):
        #store the training data (assumes it's already normalized)
        self.training_data = training_data

    def test(self, test_instance_feature_vector):
        #predict label for test instance by finding k closest neighbors
        if not self.training_data:
            print("Warning: No training data.")
            return None

        #calculate distances to all training points
        distances = []
        for train_feature_vector, train_class_label in self.training_data:
            distance = euclidean_distance(test_instance_feature_vector, train_feature_vector)
            distances.append((distance, train_class_label))

        #sort by distance and take k closest
        distances.sort(key=lambda x: x[0])
        k_nearest = distances[:self.k]

        #extract labels from k nearest neighbors
        nearest_labels = [label for _, label in k_nearest]

        #return most common label (majority vote)
        most_common = Counter(nearest_labels).most_common(1)
        return most_common[0][0]
