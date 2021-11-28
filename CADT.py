"""
Group Number : 21

Roll Numbers : Names of members
20CS60R68    : Trapti Singh
20CS60R70    : Ram Kishor Yadav
20CS60R71    : Rohit

Project number : 1
Project code   : CADT
Project title : Car Insurance Approval using Decision Tree based Learning Model
"""

# Including necessary libraries
import pandas as nap
import math
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from sklearn import preprocessing
import warnings


warnings.filterwarnings('ignore')

# Reading data for training
train_file = nap.read_csv('train.csv')


def calc_entropy(attributes_list):
    #function for calculating entropy of an attribute
    tot_instances = len(attributes_list) * 1.0
    counter_val = Counter(a for a in attributes_list)
    probability = [a / tot_instances for a in counter_val.values()]
    return sum( [-prob*math.log(prob, 2) for prob in probability] )


def calc_information_gain(train_file, attribute_divide, second_attribute, trace=0):
    # function for calculating information gain
    tot_objects = len(train_file.index) * 1.0
    divided_data = train_file.groupby(attribute_divide)

    aggregate_data = divided_data.agg({second_attribute: [calc_entropy, lambda a: len(a) / tot_objects]})[
        second_attribute]

    aggregate_data.columns = ['Entropy', 'PropObservations']

    entropy_dash = sum(aggregate_data['Entropy'] * aggregate_data['PropObservations'])
    previous_entropy = calc_entropy(train_file[second_attribute])
    return previous_entropy - entropy_dash


def id3_algorithm(train_file, second_attribute, complete_attributes):
    # Building tree from id3 algorithm
    counter_val = Counter(a for a in train_file[second_attribute])

    if len(counter_val) == 1:
        return next(iter(counter_val))

    elif train_file.empty or (not complete_attributes):
        return dc
    else:
        final_gain = [calc_information_gain(train_file, first_attribute, second_attribute) for first_attribute in complete_attributes]
        dc = max(counter_val.keys())
        best_index = final_gain.index(max(final_gain))
        best_attr = complete_attributes[best_index]
        att_left = [i for i in complete_attributes if i != best_attr]
        the_tree = {best_attr: {}}

        for abute_num, part_of_data in train_file.groupby(best_attr):
            subset_of_tree = id3_algorithm(part_of_data,
                          second_attribute,
                          att_left)
            the_tree[best_attr][abute_num] = subset_of_tree
        return the_tree


def gini_index(train_file,complete_attributes,second_attribute):
    # Function for calculation of gini index
    n_instances = float(sum([len(d) for d in train_file]))
    
    gini = 0.0
    for group in train_file:
        size = float(len(group))
        if size==0:
            continue
    score = 0.0
    # according to the score of each scoring, scoring the group
    for att_val in complete_attributes:
        p = [row[-1] for row in group].count(att_val) / size
        score += p * p
    gini += (1.0 - score) * (size / n_instances)
    return gini


def gini_index_algo(train_file, second_attribute, complete_attributes):
    # Building decision tree using gini index algorithm
    dc = None
    counter_val = Counter(a for a in train_file[second_attribute])

    if len(counter_val) == 1:
        return next(iter(counter_val))

    elif train_file.empty or (not complete_attributes):
        return dc
    else:
    
        final_gain = [gini_index(train_file, first_attribute, second_attribute) for first_attribute in complete_attributes]
        dc = max(counter_val.keys())
        best_index = final_gain.index(max(final_gain))
        best_attr = complete_attributes[best_index]
        att_left = [i for i in complete_attributes if i != best_attr]
        the_tree = {best_attr: {}}

        for abute_num, part_of_data in train_file.groupby(best_attr):
            subset_of_tree = gini_index_algo(part_of_data,
                          second_attribute,
                          att_left)
            the_tree[best_attr][abute_num] = subset_of_tree
        return the_tree
# getting names of all attributes
complete_attributes = list(train_file.columns)
complete_attributes.remove('class')

# Running both algorithms on our training data
the_tree = id3_algorithm(train_file,'class',complete_attributes)
tree1= gini_index_algo(train_file,'class',complete_attributes)


# function gor classification of data
def give_classification(occurence, the_tree, default='None'):
    next_attr = next(iter(the_tree))
    if occurence[next_attr] in the_tree[next_attr].keys():
        output = the_tree[next_attr][occurence[next_attr]]
        if isinstance(output, dict):
            return give_classification(occurence, output)
        else:
            return output
    else:
        return default

# Reading Test data
test_data = nap.read_csv('test.csv')
# Predict class for test data
test_data['predicted2'] = test_data.apply(give_classification, axis=1, args=(the_tree,'Yes'))
test_data['predicted3'] = test_data.apply(give_classification, axis=1, args=(tree1,'Yes'))
# calculating Accuracy
acc_info_gain =  sum(test_data['class']==test_data['predicted2'] ) / (1.0*len(test_data.index))
acc_gini_index =  sum(test_data['class']==test_data['predicted3'] ) / (1.0*len(test_data.index))

# print accuracy for Entropy and gini index model
print ('\nAccuracy of Information gain model: ' + str(acc_info_gain))
print ('\nAccuracy of Gini index model: ' + str(acc_gini_index))


test_class = test_data.values[:, 6]
test_pred_entropy = test_data.values[:, 7]
test_pred_gini = test_data.values[:, 8]


print('\n\n Classification Report for Gini index based Model:')

print(classification_report(test_class, test_pred_gini))


print('\n\n Classification Report for Entropy based Model:')

print(classification_report(test_class, test_pred_entropy))


le = preprocessing.LabelEncoder()
train_file = train_file.apply(le.fit_transform)

X = train_file.values[:, 0:6]
Y = train_file.values[:, 6]
X_train, X_test, y_train, y_test = train_test_split( X, Y, test_size = 0.2, random_state = 100)

# building the tree using Sklearn
dtc_entropy = DecisionTreeClassifier(criterion = "entropy",random_state = 100,
 max_depth=100, min_samples_leaf=5)
dtc_entropy.fit(X_train, y_train)




# Calculating accuracy for sklearn model
def cal_accuracy(y_test, y_pred):

    print("Accuracy : ",
          accuracy_score(y_test, y_pred) * 100)



print("\n\nResults for Entropy model using sklearn:")
# Prediction using entropy
y_pred_en = dtc_entropy.predict(X_test)
cal_accuracy(y_test, y_pred_en)