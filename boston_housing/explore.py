import numpy as np
import pandas as pd
from scipy import stats


import sklearn.cross_validation
from sklearn.metrics import r2_score

# Load the Boston housing dataset
data = pd.read_csv('housing.csv')
prices = data['MEDV']
features = data.drop('MEDV', axis=1)

target = data['MEDV']
target.min()


def test(y_true, y_predict):
    slope, intercept, r_value, p_value, std_err = stats.linregress(y_true, y_predict)
    print(r2_score(y_true, y_predict))
    print(slope, intercept, r_value, p_value, std_err)
    print(performance_metric(y_true, y_predict), ' -> ', r_value**2)
    print ''


def performance_metric(y_true, y_predict):
    """ Calculates and returns the performance score between
        true and predicted values based on the metric chosen. """

    sstot = 0
    ssres = 0
    for i in range(len(y_true)):
        sstot += (y_true[i] - y_true.mean())**2
        ssres += (y_true[i] - y_predict[i])**2
    res = 1 - ssres/sstot
    assert res == r2_score(y_true, y_predict)
    return res

        #sspred += y_true[i] - y_true.mean()
        #          * (y_predict[i] - y_predict.mean())

    #return (score / len(y_true)) / (y_true.std() * y_predict.std())
    #return ((1/len(y_true)) * score / (y_true.std() * y_predict.std())) ** 2

if 0:
    test(np.array([1, 2, 3]), np.array([1, 2, 3]))
    test(np.array([1, 2, 3]), np.array([1.1, 1.9, 2.1]))
    test(np.array([1, 2, 3]), np.array([2, 3, 4]))
    test(np.array([1, 2, 3]), np.array([3, 2, 1]))

if 0:
    train1, test1, train2, test2 = sklearn.cross_validation.train_test_split(list(range(100)), list(range(100)), test_size=0.2, train_size=0.8, random_state=1)
    print(len(train1), len(test1))
    print(len(train2), len(test2))
    print(train1[23], train2[23])


X_train, X_test, y_train, y_test = sklearn.cross_validation.train_test_split(
     features,
     target,
     train_size=0.8,
     test_size=0.2,
     random_state=1
)
from sklearn.cross_validation import ShuffleSplit
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer
from sklearn.grid_search import GridSearchCV


def fit_model(X, y):
    """ Performs grid search over the 'max_depth' parameter for a
        decision tree regressor trained on the input data [X, y]. """

    # Create cross-validation sets from the training data
    cv_sets = ShuffleSplit(X.shape[0], n_iter=10, test_size=0.20, random_state=0)

    # TODO: Create a decision tree regressor object
    regressor = DecisionTreeRegressor()

    # TODO: Create a dictionary for the parameter 'max_depth' with a range from 1 to 10
    params = {'max_depth': range(1, 10)}

    # TODO: Transform 'performance_metric' into a scoring function using 'make_scorer'
    scoring_fnc = make_scorer(performance_metric)

    # TODO: Create the grid search object
    grid = GridSearchCV(regressor, params, cv=cv_sets, scoring=scoring_fnc)

    # Fit the grid search object to the data to compute the optimal model
    grid = grid.fit(X, y)

    # Return the optimal model after fitting the data
    return grid.best_estimator_

# Fit the training data to the model using grid search
reg = fit_model(X_train, y_train)

# Produce the value for 'max_depth'
print "Parameter 'max_depth' is {} for the optimal model.".format(reg.get_params()['max_depth'])