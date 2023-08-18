import pandas as pd
import numpy as np
from scipy.stats import pearsonr


def load_data(path, effort_label, to_drop):
    if path.endswith('csv'):
        dataset = pd.read_csv(path).drop(to_drop, axis=1).replace(-1, np.nan).dropna()
    elif path.endswith('xlsx'):
        dataset = pd.read_excel(path).drop(to_drop, axis=1).replace(-1, np.nan).dropna()
    else:
        return None

    train = dataset.iloc[:-1]
    # print("Trian: \n", train)
    test  = dataset.iloc[-1]
    # print("Test: \n", test)
    train_X = train.drop(effort_label, axis=1)
    # print("Trian_X: \n", train_X)
    # print(effort_label)
    test_X = test.drop(effort_label)
    train_y = train[effort_label]
    # print("Trian_Y: \n", train_y)
    test_y = test[effort_label]

    return (train_X, train_y, test_X, test_y)

def interval01(train_X, test_X):
    max_X = np.max(train_X, axis=0)
    min_X = np.min(train_X, axis=0)

    train_X_adj = (train_X - min_X) / (max_X - min_X)
    test_X_adj = (test_X - min_X) / (max_X - min_X)

    return (train_X_adj, test_X_adj)

def calculate_nn(train_X, test_X, categorical_label):

    train_X_adj, test_X_adj = interval01(train_X, test_X)

    numerical_distance = (train_X_adj.drop(categorical_label, axis=1) - test_X_adj.drop(categorical_label)) ** 2
    categorical_distance = (1*(train_X_adj[categorical_label] == test_X_adj[categorical_label]))
    euc_distance = np.sqrt(np.sum(pd.concat([numerical_distance, categorical_distance], axis=1), axis=1)/np.shape(train_X)[0])
    rank = np.argsort(euc_distance).values
    return rank

def uavg(rank, train_y, k):
    estimate_effort = np.mean(train_y[rank[:k]])
    return estimate_effort

def irwm(rank, train_y, k):
    estimate_effort = np.sum((list(range(k,0,-1)) * train_y[rank[:k]])/np.sum(range(k+1)))
    return estimate_effort

def lsa(rank, train_y, k, train_X, test_X, size_label):
    software_size_train = train_X[size_label]
    software_size_test = test_X[size_label]
    estimate_effort = np.mean(train_y[rank[:k]]/software_size_train[rank[:k]]) * software_size_test
    return estimate_effort

def rtm(rank, train_y, k, train_X, test_X, categorical_label, size_label, group_label):
    software_size_train = train_X[size_label]
    productivity_train = train_y / software_size_train

    software_size_test = test_X[size_label]
    group_test = test_X[group_label].iloc[0]

    M = productivity_train.loc[(train_X[group_label] == group_test).values].mean()

    all_analogues_productivity = productivity_train * 0
    for i in train_X.index:
        analogues_train = train_X.drop([i])
        analogues_test = train_X.loc[i]
        all_analogues_productivity.loc[i] = productivity_train.iloc[calculate_nn(analogues_train, analogues_test, categorical_label)[0]]

    r, _ = pearsonr(productivity_train.loc[(train_X[group_label] == group_test).values], all_analogues_productivity.loc[(train_X[group_label] == group_test).values],)

    initial_predict = np.mean(train_y[rank[:k]])
    nearest_size = np.mean(software_size_train[rank[:k]])
    nearest_prod = initial_predict / nearest_size

    estimate_effort = software_size_test * (nearest_prod + (M - nearest_prod) * (1 - np.abs(r)))
    
    return estimate_effort


if __name__ == '__main__':
    k=3

    path ='albrecht.xlsx'
    effort_label='Effort'
    size_label='AdjFP'
    categorical_label = []
    group_label = ['Inquiry']
    to_drop = ['FPAdj', 'RawFP']

    train_X, train_y, test_X, test_y = load_data(path, effort_label, to_drop)
    rank = calculate_nn(train_X, test_X, categorical_label)

    estimate_effort_uavg = uavg(rank, train_y, k)
    estimate_effort_irwm = irwm(rank, train_y, k)
    estimate_effort_lsa = lsa(rank, train_y, k, train_X, test_X, size_label)
    estimate_effort_rtm = rtm(rank, train_y, k, train_X, test_X, categorical_label, size_label, group_label)

    print('With K=3')  
    print({'actual': test_y, 'uavg': estimate_effort_uavg, 'irwm': estimate_effort_irwm, 'lsa': estimate_effort_lsa, 'rtm': estimate_effort_rtm})

    err_uavg = np.abs(estimate_effort_uavg-test_y)
    err_irwm = np.abs(estimate_effort_irwm-test_y)
    err_lsa = np.abs(estimate_effort_lsa-test_y)
    err_rtm = np.abs(estimate_effort_rtm-test_y)

    print('Error With K=3')  
    print({'uavg err': err_uavg, 'irwm err': err_irwm, 'lsa err': err_lsa, 'rtm err': err_rtm})

    print("\n")

    k = 5

    estimate_effort_uavg = uavg(rank, train_y, k)
    estimate_effort_irwm = irwm(rank, train_y, k)
    estimate_effort_lsa = lsa(rank, train_y, k, train_X, test_X, size_label)
    estimate_effort_rtm = rtm(rank, train_y, k, train_X, test_X, categorical_label, size_label, group_label)

    print('With K=5')  
    print({'actual': test_y, 'uavg': estimate_effort_uavg, 'irwm': estimate_effort_irwm, 'lsa': estimate_effort_lsa, 'rtm': estimate_effort_rtm})

    err_uavg = np.abs(estimate_effort_uavg-test_y)
    err_irwm = np.abs(estimate_effort_irwm-test_y)
    err_lsa = np.abs(estimate_effort_lsa-test_y)
    err_rtm = np.abs(estimate_effort_rtm-test_y)

    print('Error With K=5')  
    print({'uavg err': err_uavg, 'irwm err': err_irwm, 'lsa err': err_lsa, 'rtm err': err_rtm})