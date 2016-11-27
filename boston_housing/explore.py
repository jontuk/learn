import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_validation import ShuffleSplit
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

test(np.array([1, 2, 3]), np.array([1, 2, 3]))
test(np.array([1, 2, 3]), np.array([1.1, 1.9, 2.1]))
test(np.array([1, 2, 3]), np.array([2, 3, 4]))
test(np.array([1, 2, 3]), np.array([3, 2, 1]))

