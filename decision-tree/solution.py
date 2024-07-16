import csv, math, argparse
from collections import Counter

class Node:
    def __init__(self, feature=None, subtrees=None, most_common_class=None, is_leaf=False):
        self.feature = feature
        self.subtrees = subtrees or {}
        self.is_leaf = is_leaf
        self.most_common_class = most_common_class
        self.default_class = None

class Leaf:
    def __init__(self, value):
        self.value = value

class ID3:
    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self.tree = None

    def load_data(self, filepath):
        with open(filepath, 'r') as file:
            reader = csv.reader(file)
            header = next(reader)
            data = [row for row in reader]

        return header, data

    def fit(self, data, parent_data, features, depth=0):
        # print("New fit call. Parameters:", data, features)
        if len(data) == 0: 
            v = self.most_common_outcome(parent_data)
            return Leaf(v)
        
        v = self.most_common_outcome(data) 
        #print("Most common outcome: ", v)

        self.default_class = v

        if len(set(row[-1] for row in data)) == 1 or len(features) == 0 or depth == self.max_depth: # If all outcomes are the same or there are no more features, return the outcome
            return Leaf(v)

        # Find the most discriminative feature
        most_discriminative_feature = self.max_gain_argument(data, features)
        print("Most discriminative feature: ", most_discriminative_feature)
        
        feature_index = features[most_discriminative_feature]
        subtrees = {}

        # Create a subtree for each value of the most discriminative feature
        for value in sorted(set(row[feature_index] for row in data)):
            subset = [row for row in data if row[feature_index] == value] # Create a subset of the data for the current value
            new_features = {f: i for f, i in features.items() if f != most_discriminative_feature} # Remove the most discriminative feature from the features
            subtree = self.fit(subset, data, new_features, depth + 1) # Recursively create a subtree
            subtrees[(most_discriminative_feature, value)] = subtree  # Tuple is the key because of the possibility that different features can have the same value which would cause a collision
        
        return Node(most_discriminative_feature, subtrees, most_common_class=v, is_leaf=False)


        
    # Find the most common outcome in the data   
    def most_common_outcome(self, data):
        class_count = Counter([row[-1] for row in data]) # Create a dictionary that counts the frequency of different outcomes
        # Find and return the most common outcome, if there are multiple, return the alphabetically first one
        print("Class count: ", class_count)
        return min(class_count, key=lambda x: (-class_count[x], x))
            

    # Find the feature that gives the most information gain    
    def max_gain_argument(self, data, features):
        entropy = self.get_entropy(data)
        gains = {feature: entropy - self.single_feature_entropy(data, features[feature]) for feature in features}
        print("Gains: ", gains)
        # Find the feature with the highest information gain, if there are multiple, return the alphabetically first one
        return min(gains, key=lambda x: (-gains[x], x))
    
    # Calculate the entropy of the data
    def get_entropy(self, data): 
        class_count = Counter([row[-1] for row in data])
        return sum(-count/len(data) * math.log2(count/len(data)) for count in class_count.values()) # Sum the entropy of each count in the class_count dictionary
    
    # Calculate the entropy of a single feature
    def single_feature_entropy(self, data, feature_index):
        feature_values = set(row[feature_index] for row in data)
        entropy = 0
        for value in feature_values:
            subset = [row for row in data if row[feature_index] == value]
            entropy += len(subset) / len(data) * self.get_entropy(subset)
        return entropy
    
    def print_tree(self):
        branches = []
        self.collect_branches(self.tree, [], branches, 1)
        print("[BRANCHES]:")
        for branch in branches:
            print(' '.join(branch))
    
    def collect_branches(self, node, path, branches, depth):
        if isinstance(node, Leaf):
            # Add path and value of the leaf node to the list of branches
            branches.append(path + [str(node.value)])
        elif isinstance(node, Node):
            for value, subtree in node.subtrees.items():
                # Recursively collect branches for each subtree
                branch_path = path + [f"{depth}:{node.feature}={value[1]}"]
                self.collect_branches(subtree, branch_path, branches, depth + 1)

    def predict_and_evaluate(self, test_data, features):
        actual_labels = [row[-1] for row in test_data]
        predictions = []
        
        # Make a prediction for each row in the test data
        for row in test_data:
            prediction = self.predict(row, features)
            predictions.append(prediction)

        #print(actual_labels)
        #print(predictions)
        
        # Create a list of unique labels and a dictionary that maps labels to indices
        unique_labels = sorted(set([label for label in actual_labels + predictions if label is not None]))
        label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
        
        # Initialize confusion matrix
        confusion_matrix = [[0 for _ in unique_labels] for _ in unique_labels]
        
        # Fill the confusion matrix
        for actual, predicted in zip(actual_labels, predictions):
            confusion_matrix[label_to_index[actual]][label_to_index[predicted]] += 1
        
        # Print results
        print("[PREDICTIONS]:", " ".join(predictions))
        # Count the number of correct predictions and calculate accuracy
        print("[ACCURACY]:", format(sum(1 for i, j in zip(actual_labels, predictions) if i == j) / len(actual_labels), '.5f'))
        # Call the function that prints the confusion matrix
        self.print_confusion_matrix(confusion_matrix, unique_labels)
    
    def print_confusion_matrix(self, matrix, labels):
        print("[CONFUSION_MATRIX]:")
        for row in matrix:
            print(" ".join(map(str, row)))

    def predict(self, row, features):
        curr = self.tree
        while isinstance(curr, Node):
            feature_index = features[curr.feature]
            value = row[feature_index]
            curr = curr.subtrees.get((curr.feature, value), curr.most_common_class) # Default to None if no subtree matches
        return curr.value if isinstance(curr, Leaf) else self.default_class
        
        

def main(): 
    parser = argparse.ArgumentParser(description='ID3 decision tree.')
    parser.add_argument("train_data", type=str, help="Path to data for training")
    parser.add_argument("test_data", type=str, help="Path to data for testing")
    parser.add_argument("limit", type=int, nargs='?', help="Limit for tree depth")

    args = parser.parse_args()

    model = ID3(args.limit)
    header, data = model.load_data(args.train_data)
    features = {feature: i for i, feature in enumerate(header[:-1])}
    model.tree = model.fit(data, data, features)
    model.print_tree()

    test_header, test_data = model.load_data(args.test_data)
    model.predict_and_evaluate(test_data, features)


    

if __name__ == "__main__":
    main()