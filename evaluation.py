from sklearn.metrics import classification_report
import pandas as pd
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

compute_calssification_report_mock()
