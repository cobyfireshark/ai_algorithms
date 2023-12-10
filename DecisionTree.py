# decision_tree/decision_tree.py
import os
import util
import logging

from random import seed
from random import randrange

class DecisionTree:

    def __init__(self, n_folds, max_depth, min_size):
        self.n_folds = n_folds
        self.max_depth = max_depth
        self.min_size = min_size

    # Split a dataset into k folds
    def cross_validation_split(self, dataset):
        logging.info("cross_validation_split()")
        dataset_split = list()
        dataset_copy = list(dataset)
        fold_size = int(len(dataset) / self.n_folds)
        for i in range(self.n_folds):
            fold = list()
            while len(fold) < fold_size:
                index = randrange(len(dataset_copy))
                fold.append(dataset_copy.pop(index))
            dataset_split.append(fold)
        return dataset_split

    # Calculate accuracy percentage
    def accuracy_metric(self, actual, predicted):
        correct = 0
        for i in range(len(actual)):
            if actual[i] == predicted[i]:
                correct += 1
        return correct / float(len(actual)) * 100.0

    # Evaluate an algorithm using a cross validation split
    def evaluate_algorithm(self, dataset, algorithm, *args):
        logging.info("evaluate_algorithm()")
        folds = self.cross_validation_split(dataset)
        logging.info(f"folds: {util.get_loggable_json(folds)}")
        scores = list()
        for fold in folds:
            train_set = list(folds)
            train_set.remove(fold)
            train_set = sum(train_set, [])
            test_set = list()
            for row in fold:
                row_copy = list(row)
                test_set.append(row_copy)
                row_copy[-1] = None

            logging.info(f"train_set: {util.get_loggable_json(train_set)}")
            logging.info(f"test_set: {util.get_loggable_json(test_set)}")
            logging.info(f"Calling {algorithm} with args (train_set, test_set)")
            predicted = algorithm(train_set, test_set, *args)
            actual = [row[-1] for row in fold]
            accuracy = self.accuracy_metric(actual, predicted)
            scores.append(accuracy)
        return scores

    # Split a dataset based on an attribute and an attribute value
    def test_split(self, index, value, dataset):
        left, right = list(), list()
        for row in dataset:
            if row[index] < value:
                left.append(row)
            else:
                right.append(row)
        return left, right

    # Calculate the Gini index for a split dataset
    def gini_index(self, groups, classes, detailed_output=False):
        
        n_instances = float(sum([len(group) for group in groups])) # count all samples at split point
        
        dataset_gini = 0.0 # sum weighted Gini index for each group
        class_scores = {}
        group_ginis = {}

        for group_id, group in enumerate(groups):
            size = float(len(group))
            # avoid divide by zero
            if size == 0:
                continue

            group_class_scores = {}
            score = 0.0
            # score the group based on the score for each class
            for class_val in classes:
                proportion = [row[-1] for row in group].count(class_val) / size
                score += proportion * proportion
                group_class_scores[class_val] = {'proportion': proportion, 'score': proportion * proportion}
        
            group_gini = 1.0 - score
            
            dataset_gini += group_gini * (size / n_instances)
            group_ginis[f"group_{group_id}"] = group_gini
            class_scores[f"group_{group_id}"] = group_class_scores
        
        if not detailed_output:
            return dataset_gini
        else:
            return dataset_gini, group_ginis, class_scores

    # Select the best split point for a dataset
    def get_split(self, dataset):
        class_values = list(set(row[-1] for row in dataset))
        b_index, b_value, b_score, b_groups = 999, 999, 999, None
        for index in range(len(dataset[0]) - 1):
            for row in dataset:
                groups = self.test_split(index, row[index], dataset)
                gini = self.gini_index(groups, class_values)
                if gini < b_score:
                    b_index, b_value, b_score, b_groups = index, row[
                        index], gini, groups
        return {'index': b_index, 'value': b_value, 'groups': b_groups}

    # Create a terminal node value
    def to_terminal(self, group):
        outcomes = [row[-1] for row in group]
        return max(set(outcomes), key=outcomes.count)

    # Create child splits for a node or make terminal
    def split(self, node, depth):
        logging.info("split()")
        left, right = node['groups']
        del (node['groups'])
        # check for a no split
        if not left or not right:
            node['left'] = node['right'] = self.to_terminal(left + right)
            return
        # check for max depth
        if depth >= self.max_depth:
            node['left'], node['right'] = self.to_terminal(left), self.to_terminal(right)
            return
        # process left child
        if len(left) <= self.min_size:
            node['left'] = self.to_terminal(left)
        else:
            node['left'] = self.get_split(left)
            self.split(node['left'], depth + 1)
        # process right child
        if len(right) <= self.min_size:
            node['right'] = self.to_terminal(right)
        else:
            node['right'] = self.get_split(right)
            self.split(node['right'], depth + 1)

    # Build a decision tree
    def build_tree(self, train):
        logging.info("build_tree()")
        root = self.get_split(train)
        logging.info(f"root: {util.get_loggable_json(root)}")
        self.split(root, 1)
        return root

    # Make a prediction with a decision tree
    def predict(self, node, row):
        if row[node['index']] < node['value']:
            if isinstance(node['left'], dict):
                return self.predict(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict(node['right'], row)
            else:
                return node['right']

    # Classification and Regression Tree Algorithm
    def decision_tree_algorithm(self, train, test):
        logging.info("decision_tree_algorithm()")
        tree = self.build_tree(train)
        predictions = list()
        for row in test:
            prediction = self.predict(tree, row)
            predictions.append(prediction)
        return (predictions)


def main():
    # Initialize logging
    log_path = os.path.join("var", "log", "decision_tree", "decision_tree.log")
    util.initialize_logging(False, log_path)
    logging.info(f"Logging initialized, output file: {log_path}")

    # Apply CART algorithm to Bank Note dataset
    seed(1)

    dataset_path = os.path.join("data_banknote_authentication", "data_banknote_authentication.csv")
    logging.info(f"Loading dataset {dataset_path}")
    dataset = util.load_csv(dataset_path, has_header=True)
    logging.info(f"dataset: {util.get_loggable_json(dataset)}")
    # convert string attributes to integers
    for i in range(len(dataset[0])):
        util.string_column_to_float(dataset, i)
    
    # evaluate algorithm
    n_folds = 5
    max_depth = 5
    min_size = 10

    logging.info("Initializing instance of DecisionTree class")
    decision_tree = DecisionTree(n_folds, max_depth, min_size)
    logging.info("Getting scores by running decision_tree functions")
    scores = decision_tree.evaluate_algorithm(dataset, decision_tree.decision_tree_algorithm)
    logging.info('Scores: %s' % scores)
    logging.info('Mean Accuracy: %.3f%%' % (sum(scores) / float(len(scores))))

if __name__ == "__main__":
    main()
