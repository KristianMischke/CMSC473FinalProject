from sklearn.metrics import classification_report
import pandas as pd
from numpy import array
from sklearn.model_selection import KFold
'''
Install packages from requirements.txt
pip3 install -r requirements.txt
'''

# Computes accuracy, precision, recall, and f1-score
# dataset: mock data from Assignment 3 - Q2 part D
# Actual =    [C, C, A, C, C, C, C, C, B, A, C, C, C]
# Predicted = [C, C, C, C, C, C, C, C, B, A, A, C, C].
# Represents: A = 1, B = 2, C = 3
def compute_calssification_report_mock():
    res = pd.DataFrame(
        [[3, 3],
         [3, 3],
         [1, 3],
         [3, 3],
         [3, 3],
         [3, 3],
         [3, 3],
         [3, 3],
         [2, 2],
         [1, 1],
         [3, 1],
         [3, 3],
         [3, 3]], columns=['Expected', 'Predicted'])

    report = classification_report(res['Expected'], res['Predicted'])
    print(res)
    print(report)

# compute_calssification_report_mock()


def compute_k_fold_cross_validation(K=10):
    print('compute_k_fold_cross_validation')
    X = array([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.1, 0.2], [0.3, 0.4], [0.5, 0.6], [0.7, 0.8]])
    y = array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
    kfold = KFold(n_splits=10, shuffle=True, random_state=None)
    for train_indx, test_indx in kfold.split(X):
        print("TRAIN:", X[train_indx], "TEST:", X[test_indx])
        y_train, y_test = y[train_indx], y[test_indx]

compute_k_fold_cross_validation()