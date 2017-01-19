
# Import libraries
import numpy as np
import pandas as pd
from time import time
from sklearn.metrics import f1_score

student_data = pd.read_csv("student-data.csv")

# Extract feature columns
feature_cols = list(student_data.columns[:-1])

# Extract target column 'passed'
target_col = student_data.columns[-1]

# Separate the data into feature data and target data (X_all and y_all, respectively)
X_all = student_data[feature_cols]
y_all = student_data[target_col]


def preprocess_features(X):
    ''' Preprocesses the student data and converts non-numeric binary variables into
        binary (0/1) variables. Converts categorical variables into dummy variables. '''

    # Initialize new output DataFrame
    output = pd.DataFrame(index=X.index)

    # Investigate each feature column for the data
    for col, col_data in X.iteritems():

        # If data type is non-numeric, replace all yes/no values with 1/0
        if col_data.dtype == object:
            col_data = col_data.replace(['yes', 'no'], [1, 0])

        # If data type is categorical, convert to dummy variables
        if col_data.dtype == object:
            # Example: 'school' => 'school_GP' and 'school_MS'
            col_data = pd.get_dummies(col_data, prefix=col)

            # Collect the revised columns
        output = output.join(col_data)

    return output


X_all = preprocess_features(X_all)

#------------------------------------------

# TODO: Import any additional functionality you may need here
import sklearn.cross_validation

# TODO: Set the number of training points
num_train = 300

# Set the number of testing points
num_test = X_all.shape[0] - num_train

# TODO: Shuffle and split the dataset into the number of training and testing points above

train_idxs, test_idxs \
    = sklearn.cross_validation.ShuffleSplit(n=X_all.shape[0], test_size=num_test, n_iter=1, random_state=1)\
    .__iter__().next()

X_train = X_all.take(train_idxs)
X_test = X_all.take(test_idxs)
y_train = y_all.take(train_idxs)
y_test = y_all.take(test_idxs)

# Show the results of the split
print("Training set has {} samples.".format(X_train.shape[0]))
print("Testing set has {} samples.".format(X_test.shape[0]))

#----------------------------------------------------


def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()

    # Print the results
    print("Trained model in {:.4f} seconds".format(end - start))


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print("Made predictions in {:.4f} seconds.".format(end - start))
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))


def train_predict_md(n, i, clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    if i == 0:
        print("\n** Classifer {} - {}**  \n".format(n, clf.__class__.__name__,))
        print('| Training Set Size | Training Time | Prediction Time (test) | F1 Score (train) | F1 Score (test) |')
        print('| :---------------: | :---------------------: | :--------------------: | :--------------: | :-------------: |')

    #print('| %s | %s | %s | %s | %s |', len(X_train), train_time)

    len(X_train)
    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print("F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train)))
    print("F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test)))



# TODO: Import the three supervised learning models from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression

# TODO: Initialize the three models
clf_A = KNeighborsClassifier()
clf_B = SGDClassifier()
clf_C = LogisticRegression()
clfs = (clf_A, clf_B, clf_C)

trainings = list()
# TODO: Set up the training set sizes
X_train_100 = X_train[:100]
y_train_100 = y_train[:100]
trainings += [(X_train_100, y_train_100)]

X_train_200 = X_train[:200]
y_train_200 = y_train[:200]
trainings += [(X_train_200, y_train_200)]

X_train_300 = X_train[:300]
y_train_300 = y_train[:300]
trainings += [(X_train_300, y_train_300)]

# TODO: Execute the 'train_predict' function for each classifier and each training set size
for n, clf in enumerate(clfs):
    for i, (X_train, y_train) in enumerate(trainings):
        train_predict_md(n, i, clf, X_train, y_train, X_test, y_test)
