from analogy import load_data
from sklearn import linear_model, tree, svm


import pandas as pd
import numpy as np

if __name__ == '__main__':

    path='02.desharnais.csv'
    effort_label='Effort'
    size_label='PointsAjust'
    categorical_label = ['Language']
    to_drop = ['id', 'PointsNonAdjust', 'Adjustment', 'YearEnd', 'Project']

    # path='resource/albrecht.xlsx'
    # effort_label='Effort'
    # size_label='AdjFP'
    # categorical_label = []
    # to_drop = ['FPAdj', 'RawFP']

    train_X, train_y, test_X, test_y = load_data(path, effort_label, to_drop)

    clf = tree.DecisionTreeRegressor()
    clf = clf.fit(train_X, train_y)
    estimate_effort_dt = clf.predict(pd.DataFrame(test_X).T)

    clf = linear_model.LinearRegression()
    clf = clf.fit(train_X, train_y)
    estimate_effort_ols = clf.predict(pd.DataFrame(test_X).T)

    clf = svm.SVR()
    clf = clf.fit(train_X, train_y)
    estimate_effort_svr = clf.predict(pd.DataFrame(test_X).T)

    print({'actual': test_y, 'dt': estimate_effort_dt, 'ols': estimate_effort_ols, 'svr': estimate_effort_svr})

    err_dt = np.abs(estimate_effort_dt - test_y)
    err_ols = np.abs(estimate_effort_ols - test_y)
    err_svr = np.abs(estimate_effort_svr - test_y)

    print({'dt err': err_dt, 'ols err': err_ols, 'svr err': err_svr})