#loads datasets from files and dpes feature normalization
def load_dataset(filepath):
    dataset = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:  #skip empty lines
                continue
            parts = line.split()
            try:
                class_label = float(parts[0])
                feature_vector = [float(x) for x in parts[1:]]
                dataset.append((feature_vector, class_label))
            except ValueError as e:
                print(f"Warning: Skipping line due to parsing error: '{line}' - {e}")
    return dataset

#apply min-max normalization to features
def normalize_features(dataset):
    if not dataset:
        return []

    num_features = len(dataset[0][0])
    
    #find min and max for each feature column
    min_vals = [float('inf')] * num_features
    max_vals = [float('-inf')] * num_features

    for feature_vector, _ in dataset:
        for i in range(num_features):
            if feature_vector[i] < min_vals[i]:
                min_vals[i] = feature_vector[i]
            if feature_vector[i] > max_vals[i]:
                max_vals[i] = feature_vector[i]

    normalized_dataset = []
    for feature_vector, class_label in dataset:
        normalized_vector = []
        for i in range(num_features):
            feature_value = feature_vector[i]
            range_val = max_vals[i] - min_vals[i]
            if range_val == 0:
                normalized_value = 0.0 #handle cases where feature has a constant value (no variance)
            else:
                normalized_value = (feature_value - min_vals[i]) / range_val
            normalized_vector.append(normalized_value)
        normalized_dataset.append((normalized_vector, class_label))
        
    return normalized_dataset