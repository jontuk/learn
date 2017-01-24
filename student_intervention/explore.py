
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
    print "Trained model in {:.4f} seconds".format(end - start)


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    end = time()

    # Print and return results
    print "Made predictions in {:.4f} seconds.".format(end - start)
    return f1_score(target.values, y_pred, pos_label='yes')


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print "Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train))

    # Train the classifier
    train_classifier(clf, X_train, y_train)

    # Print the results of prediction for both training and testing
    print "F1 score for training set: {:.4f}.".format(predict_labels(clf, X_train, y_train))
    print "F1 score for test set: {:.4f}.".format(predict_labels(clf, X_test, y_test))


# TODO: Import the three supervised learning models from sklearn
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

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
if 0:
    for n, clf in enumerate(clfs):
        for i, (X_train, y_train) in enumerate(trainings):
            train_predict(clf, X_train, y_train, X_test, y_test)

#--------------

# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV

# TODO: Create the parameters list you wish to tune
# parameters = dict(loss=('hinge',
#                         'log',
#                         'modified_huber',
#                         'squared_hinge',
#                         'perceptron',
#                         'squared_loss',
#                         'huber',
#                         'epsilon_insensitive',
#                         'squared_epsilon_insensitive'),
#                   penalty=('l2', 'l1', 'elasticnet'),
#                   fit_intercept=(True, False),
#                   n_iter=[10**i for i in range(3)],
#                   random_state=(1,),
#                   alpha=[0.00005 * 2**i for i in range(5)])

# TODO: Create the parameters list you wish to tune
parameters = dict(kernel=('linear', 'poly', 'rbf', 'sigmoid'),
                  degree=(2, 3, 4),
                  C=(.5, .8, .9, .99, 1, 1.01, 1.1, 1.2, 1.5),
                  gamma=(.1, .3, .5, .7, .9, .11, .13, .15),
                  tol=(0.0005, 0.001, 0.002),
                )

# parameters = dict(n_neighbors=range(1, 50),
#                   weights=('uniform', 'distance'),
#                   n_jobs=(1,),
#                   p=(1, 2, 3),
#                   algorithm=('ball_tree', 'kd_tree', 'brute'))

# TODO: Initialize the classifier
clf = SVC()

# TODO: Make an f1 scoring function using 'make_scorer'
def f1_score_with_label(*args, **kwargs):
    kwargs['pos_label'] = 'yes'
    return f1_score(*args, **kwargs)

f1_scorer = make_scorer(f1_score_with_label)

# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf, parameters, scoring=f1_scorer, verbose=1, n_jobs=4, )

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(X_train, y_train)
print(grid_obj)

# Get the estimator
clf = grid_obj.best_estimator_

# Report the final F1 score for training and testing after parameter tuning
print "Tuned model has a training F1 score of {:.4f}.".format(predict_labels(clf, X_train, y_train))
print "Tuned model has a testing F1 score of {:.4f}.".format(predict_labels(clf, X_test, y_test))

print grid_obj.grid_scores_[0]
print grid_obj.grid_scores_[0][1]

for d in sorted(grid_obj.grid_scores_, lambda x1, x2: int(np.sign(x1[1] - x2[1]))):
    print d



#pd.DataFrame(grid_obj.cv_results_)
#
# The advantages of support vector machines are:
#
#         Effective in high dimensional spaces.
#         Still effective in cases where number of dimensions is greater than the number of samples.
#         Uses a subset of training points in the decision function (called support vectors), so it is also memory efficient.
#         Versatile: different Kernel functions can be specified for the decision function. Common kernels are provided, but it is also possible to specify custom kernels.
#
# The disadvantages of support vector machines include:
#
#         If the number of features is much greater than the number of samples, the method is likely to give poor performances.
#         SVMs do not directly provide probability estimates, these are calculated using an expensive five-fold cross-validation (see Scores and probabilities, below).
#
