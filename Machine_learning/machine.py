
from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score, precision_score, adjusted_rand_score,roc_auc_score, balanced_accuracy_score, recall_score

import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import random,os, torch
import xlsxwriter
from machine_learning import Multi_Classfier


def load_gene_dict():
    gene_file = "../Dataset/gene_features/TopGene_300.txt"
    Data = pd.read_table(gene_file, sep='\t', header='infer', names=None, index_col=None, dtype=None, engine=None,
                         nrows=None)
    return Data
def load_rsicp_dict():
    file_path = "../hic_features/RLDCP_1.npy"
    Data = np.load(file_path, allow_pickle=True).item()  # 返回的长度为细胞数量
    return Data

def load_multi_dict():
    gene_file = "../Dataset/gene_features/hic_multi_nor.txt"
    Data = pd.read_table(gene_file, sep='\t', header='infer', names=None, index_col=None, dtype=None, engine=None,
                         nrows=None)
    return Data

def main():
    path = "../cell_cycle.txt"
    cell_inf = pd.read_table(path, sep='\t', header='infer', names=None, index_col=None, dtype=None, engine=None,
                             nrows=None)
    cell_inf = cell_inf.sort_values(by='Cellcycle', ascending=True)
    X_index = np.array(cell_inf.values.tolist())[:, 0:1]
    Y = []
    # num用于统计每种类型细胞的数量
    num = {0: 0, 1: 0, 2: 0, 3: 0}
    label = np.array(cell_inf.values.tolist())[:, 1:]
    for l in label:
        if l == 'G1':
            Y.append(0)
            num[0] = num[0] + 1
        elif l == "Early-S":
            Y.append(1)
            num[1] = num[1] + 1
        elif l == "Mid-S":
            Y.append(2)
            num[2] = num[2] + 1
        elif l == "Late-S/G2":
            Y.append(3)
            num[3] = num[3] + 1
    Y = np.array(Y) # 6288
    print(num)
    # alpha用于计算focal_loss
    # 每个类别对应的alpha=该类别出现频率的倒数
    alpha = []
    for value in num.values():
        ds = 1 / value
        alpha.append(ds)
    print(alpha)
    rsicp_feature = load_rsicp_dict()  # 6288 * 3001  3000个特征

    multi_feature = load_multi_dict()
    multi_feature = multi_feature.rename(columns={multi_feature.columns[0]:'cell_name'})

    gene_feature = load_gene_dict()  # 6288 * 3001  3000个特征
    print(gene_feature)
    gene_feature = gene_feature.rename(columns={gene_feature.columns[0]:'cell_name'})
    # print(gene_feature)

    data = [rsicp_feature,multi_feature,gene_feature]

    test_seed = [7396]
    row = 0
    file = 'machine.xlsx'
    workbook = xlsxwriter.Workbook(file)
    worksheet1 = workbook.add_worksheet('model')
    worksheet1.write(0, 0, '随机种子')
    worksheet1.write(0, 1, 'SVM_test_acc')
    worksheet1.write(0, 2, 'SVM_F1')
    worksheet1.write(0, 3, 'SVM_Precision')
    worksheet1.write(0, 4, 'SVM_bacc')
    worksheet1.write(0, 5, 'SVM_ari')
    worksheet1.write(0, 6, 'SVM_recall')
    worksheet1.write(0, 7, 'Log_test_acc')
    worksheet1.write(0, 8, 'Log_F1')
    worksheet1.write(0, 9, 'Log_Precision')
    worksheet1.write(0, 10, 'Log_bacc')
    worksheet1.write(0, 11, 'Log_ari')
    worksheet1.write(0, 12, 'Log_recall')
    worksheet1.write(0, 13, 'rf_test_acc')
    worksheet1.write(0, 14, 'rf_F1')
    worksheet1.write(0, 15, 'rf_Precision')
    worksheet1.write(0, 16, 'rf_bacc')
    worksheet1.write(0, 17, 'rf_ari')
    worksheet1.write(0, 18, 'rf_recall')
    for seed in test_seed:
        X_train, X_test, y_train, y_test = train_test_split(X_index, Y, test_size=0.2, random_state=seed, stratify=Y) #7396:0.638
        row = row + 1
        SVM_acc, Log_acc, rf_acc = Multi_Classfier(data, X_train, y_train, X_test, y_test, 'RSICP-MF-GENE')
        print('svm:', 'SVM_train_acc', SVM_acc[0], 'SVM_test_acc', SVM_acc[1], 'SVM_test_f1', SVM_acc[2],
              'SVM_test_precision', SVM_acc[3], 'SVM_test_ari', SVM_acc[4], 'SVM_test_bacc', SVM_acc[5])
        print('log: ', 'log_train_acc', Log_acc[0], 'log_test_acc', Log_acc[1], 'log_test_f1', Log_acc[2],
              'log_test_precision', Log_acc[3], 'log_test_ari', Log_acc[4], 'log_test_bacc', Log_acc[5])
        print('rf: ', 'rf_train_acc', rf_acc[0], 'rf_test_acc', rf_acc[1], 'rf_test_f1', rf_acc[2], 'rf_test_precision',
              rf_acc[3], 'rf_test_ari', rf_acc[4], 'rf_test_bacc', rf_acc[5])
        with open("machine_50.txt", "a") as file:
            # 写入数据
            file.write(str(seed) + ' ACC:' + str(SVM_acc[1])  + 'F1: ' + str(SVM_acc[2])
                       +'Precision:' + str(SVM_acc[2])  + '\n')
        worksheet1.write(row, 0, seed)
        worksheet1.write(row, 1, SVM_acc[1])
        worksheet1.write(row, 2, SVM_acc[2])
        worksheet1.write(row, 3, SVM_acc[3])
        worksheet1.write(row, 4, SVM_acc[4])
        worksheet1.write(row, 5, SVM_acc[5])
        worksheet1.write(row, 6, SVM_acc[6])
        worksheet1.write(row, 7, Log_acc[1])
        worksheet1.write(row, 8, Log_acc[2])
        worksheet1.write(row, 9, Log_acc[3])
        worksheet1.write(row, 10, Log_acc[4])
        worksheet1.write(row, 11, Log_acc[5])
        worksheet1.write(row, 12, Log_acc[6])
        worksheet1.write(row, 13, rf_acc[1])
        worksheet1.write(row, 14, rf_acc[2])
        worksheet1.write(row, 15, rf_acc[3])
        worksheet1.write(row, 16, rf_acc[4])
        worksheet1.write(row, 17, rf_acc[5])
        worksheet1.write(row, 18, rf_acc[6])

    workbook.close()

if __name__ == '__main__':
    main()