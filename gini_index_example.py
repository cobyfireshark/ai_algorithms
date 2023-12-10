import DecisionTree

dataset_1_classes = [0,1,2]
number_groups = len(dataset_1_classes)
print(f"dataset_1_classes:\n{dataset_1_classes}\nso number_groups={number_groups}\n")
dataset_1 = []
for k, dataset_1_class in enumerate(dataset_1_classes):
    current_group_rows = []
    for i in range(number_groups):
        current_row = ["feature_1", "feature_2", dataset_1_class]
        current_group_rows.append(current_row)
        print(f"group_{k}_row_{i}:\n{current_row}")
        print(f"current_group_rows:\n{current_group_rows}\n")

    dataset_1.append(current_group_rows)
    print(f"dataset_1:\n{dataset_1}\n")

decision_tree = DecisionTree.DecisionTree(5, 5, 10)
dataset_1_gini, group_ginis, class_scores = decision_tree.gini_index(dataset_1, dataset_1_classes, detailed_output=True)

print(f"dataset_1_gini: {dataset_1_gini}")
print(f"group_ginis:\n{group_ginis}")
print(f"class_scores:\n{class_scores}")
