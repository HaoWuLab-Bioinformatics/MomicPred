
from collections import Counter

import pandas as pd
from sklearn.metrics import f1_score, precision_score, adjusted_rand_score,roc_auc_score, balanced_accuracy_score, recall_score

from best_para_method import  CNN_1D_network
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
import random,os, torch
import xlsxwriter
from collections import Counter
from sklearn.metrics import roc_curve, auc
# from scipy import interp
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score, precision_score, adjusted_rand_score,roc_auc_score

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


    test_seed = [7396]
    row = 0
    file = 'HEG.xlsx'
    workbook = xlsxwriter.Workbook(file)
    worksheet1 = workbook.add_worksheet('model')
    worksheet1.write(0, 1, 'kernel_size')
    worksheet1.write(0, 2, 'Acc')
    worksheet1.write(0, 3, 'micro_F1')
    worksheet1.write(0, 4, 'macro_F1')
    worksheet1.write(0, 5, 'micro_Precision')
    worksheet1.write(0, 6, 'macro_Precision')
    worksheet1.write(0, 7, 'Precision')
    worksheet1.write(0, 8, 'Mcc')
    worksheet1.write(0, 9, 'F1')
    worksheet1.write(0, 10, 'ARI')
    worksheet1.write(0, 11, 'BACC')
    worksheet1.write(0, 12, 'NMI')
    worksheet1.write(0, 13, 'micro_average_ROC_curve')

    worksheet1.write(0, 15, 'auprc-AP')
    worksheet1.write(0, 16, 'roc_ovo')
    worksheet1.write(0, 17, 'roc_ovr')
    worksheet1.write(0, 18, 'test_seed')
    worksheet1.write(0, 19, 'count_label')
    worksheet1.write(0, 20, 'Recall')
    worksheet1.write(0, 21, 'Recall_micro')
    worksheet1.write(0, 22, 'Recall_macro')

    for seed in test_seed:
        X_train, X_test, y_train, y_test = train_test_split(X_index, Y, test_size=0.2, random_state=seed, stratify=Y) #7396:0.638
        row = row + 1
        linear_layer = 1
        kernel_size = 7
        # 11:  7: 0.638 5:9:
        cnn_feature = 32 # 通道数
        # 32:0.638  64:  16:
        out_feature = 64
        dp = 0.1
        lr = 0.0001
        gamma = 1
        Con_layer_gene = 2
        Con_layer_hic = 2
        Con_layer = [Con_layer_hic, Con_layer_gene]
        dp_trans, heads, block = 0.3, 5 , 4
        channel, b, eca_gamma = 2, 1, 4
        lstm_layer = 4
        lstm_para = lstm_layer
        transformer_para = [dp_trans, heads, block]
        eca_para = [channel, b, eca_gamma]
        model_para = [kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer]
        test_acc, test_label, real_label, test_result_matrix = CNN_1D_network(rsicp_feature,gene_feature,multi_feature, X_train, y_train, X_test,
                                                                              y_test, lr, model_para, alpha, gamma,
                                                                              transformer_para,eca_para, lstm_para)
        print('test_acc: ', test_acc)
        label_count = []


        for i, j in zip(test_label, real_label):
            if i == j:
                label_count.append(i) # 预测对的数目
        print("预测结果：", Counter(label_count))

        from sklearn import metrics

        micro_F1 = f1_score(real_label, test_label, average='micro')
        print("micro_F1：", micro_F1)
        macro_F1 = f1_score(real_label, test_label, average='macro')
        print("macro_F1：", macro_F1)
        micro_Precision = precision_score(real_label, test_label, average='micro')
        print("micro_Precision：", micro_Precision)
        macro_Precision = precision_score(real_label, test_label, average='macro')
        print("macro_Precision：", macro_Precision)
        Precision = metrics.precision_score(real_label, test_label, average='weighted')
        print("Precision：", Precision)
        Mcc = metrics.matthews_corrcoef(real_label, test_label)
        print("Mcc：", Mcc)
        F1 = metrics.f1_score(real_label, test_label, average='weighted')
        print("F1：", F1)
        ari = metrics.adjusted_rand_score(real_label, test_label)
        print("ARI：", ari)
        Bacc = metrics.balanced_accuracy_score(real_label, test_label)
        print("Bacc：", Bacc)
        Nmi = metrics.normalized_mutual_info_score(real_label, test_label)
        print("NMI", Nmi)
        Recall_micro = metrics.recall_score(real_label,test_label,average='micro')
        print("Recall_micro", Recall_micro)
        Recall_macro = metrics.recall_score(real_label,test_label,average='macro')
        print("Recall_macro", Recall_macro)
        Recall = metrics.recall_score(real_label,test_label,average='weighted')
        print("Recall", Recall)
        for i in range(len(test_result_matrix)):
            min_line = min(test_result_matrix[i])
            if min_line < 0:
                for j in range(len(test_result_matrix[0])):
                    test_result_matrix[i][j] = test_result_matrix[i][j] + abs(min_line)
            line_sum = sum(test_result_matrix[i])
            for j in range(len(test_result_matrix[0])):
                test_result_matrix[i][j] = test_result_matrix[i][j] / line_sum

        roc_ovr = roc_auc_score(real_label, test_result_matrix, average='macro', sample_weight=None, max_fpr=None,
                                multi_class='ovr', labels=None)
        roc_ovo = roc_auc_score(real_label, test_result_matrix, average='macro', sample_weight=None, max_fpr=None,
                                multi_class='ovo', labels=None)
        print("roc_ovo:", roc_ovo)
        print("roc_ovr:", roc_ovr)
        # print(real_label)

        # 把[1 3 0 ... 2]格式的real_label转化为下面的格式：
        # [[0,1,0,0],[0,0,0,1],[1,0,0,0]......[0,0,1,0]]
        # 也就是说类别若为1那么就用[0,1,0,0]表示
        real_label2 = []
        for i in real_label:
            if i == 0:
                real_label2.append([1, 0, 0,0])
            elif i == 1:
                real_label2.append([0, 1, 0,0])
            elif i == 2:
                real_label2.append([0, 0, 1,0])
            elif i == 3:
                real_label2.append([0, 0, 0, 1])
        # print(real_label2)

        true_label = np.array(real_label2)
        y_score = np.array(test_result_matrix)
        n_classes = 4
        # 计算每一类的ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        for i in range(n_classes):
            fpr[i], tpr[i], _ = roc_curve(true_label[:, i], y_score[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])

        # micro（方法二）
        fpr["micro"], tpr["micro"], _ = roc_curve(true_label.ravel(), y_score.ravel())
        roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

        # pr曲线
        # (1) For each class
        precision = dict()
        recall = dict()
        average_precision = dict()
        for i in range(n_classes):
            precision[i], recall[i], _ = precision_recall_curve(true_label[:, i],
                                                                y_score[:, i])

            average_precision[i] = average_precision_score(true_label[:, i],
                                                           y_score[:, i])

        # (2) A "macro-average": quantifying score on all classes jointly
        precision["macro"], recall["macro"], _ = precision_recall_curve(true_label.ravel(),
                                                                        y_score.ravel())

        average_precision["macro"] = average_precision_score(true_label, y_score,
                                                             average="macro")

        print('Average precision score, macro-averaged over all classes: {0:0.2f}'.format(average_precision["macro"]))

        worksheet1.write(row, 1, kernel_size)
        worksheet1.write(row + 1, 2, test_acc)
        worksheet1.write(row + 1, 3, micro_F1)
        worksheet1.write(row + 1, 4, macro_F1)
        worksheet1.write(row + 1, 5, micro_Precision)
        worksheet1.write(row + 1, 6, macro_Precision)
        worksheet1.write(row + 1, 7, Precision)
        worksheet1.write(row + 1, 8, Mcc)
        worksheet1.write(row + 1, 9, F1)
        worksheet1.write(row + 1, 10, ari)
        worksheet1.write(row + 1, 11, Bacc)
        worksheet1.write(row + 1, 12, Nmi)
        worksheet1.write(row + 1, 13, roc_auc["micro"])

        worksheet1.write(row + 1, 15, average_precision["macro"])
        worksheet1.write(row + 1, 16, roc_ovo)
        worksheet1.write(row + 1, 17, roc_ovr)
        worksheet1.write(row + 1, 18, seed)
        worksheet1.write(row + 1, 19, str(Counter(label_count)))
        worksheet1.write(row + 1, 20, Recall)
        worksheet1.write(row + 1, 21, Recall_micro)
        worksheet1.write(row + 1, 22, Recall_macro)

    workbook.close()

if __name__ == '__main__':
    main()