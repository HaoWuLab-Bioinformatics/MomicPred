import os, random
from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
from sklearn import svm,metrics
from torch.optim import Adam

#用于对比的机器学习方法
def str_reverse(s):
    # 倒排列表中的字符串，和下边俩改文件名的函数有关
    s = list(s)
    s.reverse()
    return "".join(s)

def replace_linetodot(S):
    # 将字符串中的横线改为点，这个是用来改文件名的
    S = str_reverse(S)
    S = S.replace('_','.',1)
    return str_reverse(S)


def generate_bin():
    f = open("../mm10.main.nochrM.chrom.sizes")
    index = {}
    resolution = 1000000
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        chr_name = chr_name
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    return index


import math
def load_gene_data(gene,idX,Y):
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = gene.loc[gene['cell_name'] == replace_linetodot(cell_name)].values[:, 1:].tolist()[0]
        value = [math.log(x) if x> 0 else 0 for x in value]
        X.append(value)
    # print('gene', X[:1])
    # gene_feature_tensor = torch.Tensor(np.array(X)).float()
    print(np.array(X).shape)

    return X,Y, np.array(X).shape[0]

def load_multi_data(gene,idX,Y):
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = gene.loc[gene['cell_name'] == replace_linetodot(cell_name)].values[:, 1:].tolist()[0]
        value = [math.log(x) if x > 0 else 0  for x in value]
        # value = [x if x > 0 else 0  for x in value]
        X.append(value)
    return X,Y, np.array(X).shape[0]

def load_rscip_data(RSICP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        cell_name = cell[0]
        rscip = []
        for chr in chr_list:
            if chr == "chrY":
                continue
            m_list = RSICP[cell_name][chr]
            result_list = [math.log(x) if x > 0 else 0 for x in m_list]
            rscip.append(result_list)
        X.append(np.concatenate(rscip).tolist())
    print(np.array(X).shape)
    return X,Y, np.array(X).shape[0]


def Multi_Classfier(Data, tr_x, tr_y, te_X, te_Y, type):
    if type == 'RSICP-MF-GENE':
        RSICP,MF, GENE = Data
        RSICP_Train_X, RSICP_Train_Y, RSICP_Train_Size = load_rscip_data(RSICP, tr_x, tr_y)
        RSICP_Test_X, RSICP_Test_Y, RSICP_Test_Size = load_rscip_data(RSICP, te_X, te_Y)
        print('RSICP over')
        MF_Train_X, MF_Train_Y, MF_Train_Size = load_multi_data(MF, tr_x, tr_y)
        MF_Test_X, MF_Test_Y, MF_Test_Size = load_multi_data(MF, te_X, te_Y)
        print('MF over')
        GENE_Train_X, GENE_Train_Y, GENE_Train_Size = load_gene_data(GENE, tr_x, tr_y)
        GENE_Test_X, GENE_Test_Y, GENE_Test_Size = load_gene_data(GENE, te_X, te_Y)
        print('GENE over')
        Train_X = np.hstack((MF_Train_X, RSICP_Train_X, GENE_Train_X))
        Train_Y = RSICP_Train_Y
        Test_X = np.hstack((MF_Test_X, RSICP_Test_X, GENE_Test_X))
        Test_Y = RSICP_Test_Y

    svm_train_acc,  svm_test_acc, svm_test_f1, svm_test_precision, svm_test_ari, svm_test_bacc, svm_test_recall = SVM(Train_X, Train_Y,  Test_X, Test_Y)
    svm = [svm_train_acc, svm_test_acc, svm_test_f1, svm_test_precision, svm_test_ari, svm_test_bacc, svm_test_recall]
    log_train_acc,  log_test_acc, log_test_f1, log_test_precision, log_test_ari, log_test_bacc, log_recall= logistic_Reg(Train_X, Train_Y, Test_X, Test_Y)
    logistic = [log_train_acc,  log_test_acc, log_test_f1, log_test_precision, log_test_ari, log_test_bacc, log_recall]
    rf_train_acc,rf_test_acc, rf_test_f1, rf_test_precision, rf_test_ari, rf_test_bacc, rf_recall = randomForest(Train_X, Train_Y, Test_X, Test_Y)
    randomFor = [rf_train_acc, rf_test_acc, rf_test_f1, rf_test_precision, rf_test_ari, rf_test_bacc, rf_recall]

    return svm, logistic, randomFor

def Search(train_model,params, Train_X,Train_Y, Test_X, Test_Y):
    # 网格寻优的训练及测试
    print('search')
    train_model = GridSearchCV(estimator=train_model, param_grid=params, cv=5)
    train_model.fit(Train_X, Train_Y)
    print('fit')
    test_label_pred = train_model.predict(Test_X)
    test_acc = metrics.accuracy_score(Test_Y, test_label_pred)
    test_f1 = metrics.f1_score(Test_Y, test_label_pred,average='micro')
    test_precision = metrics.precision_score(Test_Y, test_label_pred, average='weighted')
    test_ari = metrics.adjusted_rand_score(Test_Y, test_label_pred)
    # 加bacc
    test_bacc = metrics.balanced_accuracy_score(Test_Y, test_label_pred)
    test_recall = metrics.recall_score(Test_Y, test_label_pred, average='weighted')
    return train_model.best_score_, test_acc, test_f1, test_precision, test_ari, test_bacc, test_recall

def SVM(Train_X, Train_Y, Test_X, Test_Y):
    #SVM
    print('svm')
    train_model = svm.SVC(probability=True)
    # params = [
    #     {'kernel': ['linear'], 'C': [1]},
    #     {'kernel': ['poly'], 'C': [1], 'degree': [2]},
    #     {'kernel': ['rbf'], 'C': [1],
    #      'gamma': [0.01]}]
    params = [
        {'kernel': ['poly'], 'C': [1000], 'degree': [10]}]
    print(params)
    train_acc,  test_acc, test_f1, test_precision, test_ari, test_bacc, test_recall = Search(train_model, params, Train_X, Train_Y, Test_X, Test_Y)

    return train_acc,  test_acc, test_f1, test_precision, test_ari, test_bacc, test_recall

def logistic_Reg(Train_X, Train_Y, Test_X, Test_Y):
    #逻辑回归
    print('lr')
    train_model = LogisticRegression()
    params = [{'C': [10,100]}]
    print(params)
    train_acc, test_acc, test_f1, test_precision, test_ari, test_bacc, test_recall = Search(train_model, params, Train_X, Train_Y,Test_X, Test_Y)

    return train_acc,  test_acc, test_f1, test_precision, test_ari, test_bacc, test_recall

def randomForest(Train_X, Train_Y, Test_X, Test_Y):
    #随机森林
    print('rm')
    train_model = RandomForestClassifier()
    params = {"n_estimators":[10,50,100],"max_depth":[2,4,6,8]}
    print(params)
    train_acc,  test_acc, test_f1, test_precision, test_ari, test_bacc, test_recall = Search(train_model,params, Train_X,Train_Y, Test_X, Test_Y)

    return train_acc, test_acc, test_f1, test_precision, test_ari, test_bacc, test_recall


