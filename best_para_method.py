import os, random

import pandas as pd
import numpy as np
# from methods import RLDCPrate_bin, replace_linetodot
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch.nn as nn
# from pytorch_tools import EarlyStopping
from focal_loss_best import MultiClassFocalLossWithAlpha
import fusion_transformer
from torch.nn import functional as F


from torch.optim import Adam
resolution = 1000000
def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

def RLDCPrate_bin():
    f = open("../ECA(hic)+LSTM_Trans(rna)_融合CNN特征/mm10.main.nochrM.chrom.sizes")
    index= {}
    lines = f.readlines()
    for line in lines:
        chr_name, length = line.split()
        chr_name = chr_name
        max_len = int(int(length) / resolution)
        index[chr_name] = max_len + 1
        f.seek(0, 0)
    f.close()
    return index

def read_pair(path):
    file = open(path)
    file.readline()
    a = []
    for line in file.readlines():
        a.append(line.split())
    a = np.array(a).astype(str)
    return a

def RLDCPrate_contact_matrix(index,pair_list):
    contact_matrix = np.zeros((index, index))
    for pair in pair_list:
        bin1, bin2, num = pair
        contact_matrix[int(bin1), int(bin2)] += int(num)
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(num)
    return contact_matrix
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
    f = open("mm10.main.nochrM.chrom.sizes")
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

def seed_torch(seed=2021):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
import math
def load_rscip_data(RLDCP, idX, Y):
    index = generate_bin()
    chr_list = sorted(index.keys())
    X = []
    for cell in idX:
        cell_name = cell[0]
        rscip = []
        for chr in chr_list:
            if chr == "chrY":
                continue
            m_list = RLDCP[cell_name][chr]
            result_list = [math.log(x) if x > 0 else 0 for x in m_list]
            rscip.append(result_list)
        X.append(np.concatenate(rscip).tolist())
    print(np.array(X).shape)
    # print('SICP', X[:3])
    # print('SICP',X)
    deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    return deal_dataset, np.array(X).shape[0]

def load_gene_data(gene,idX,Y):
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = gene.loc[gene['cell_name'] == replace_linetodot(cell_name)].values[:, 1:].tolist()[0]
        value = [math.log(x) if x> 0 else 0 for x in value]
        X.append(value)
    # print('gene', X[:1])
    gene_feature_tensor = torch.Tensor(np.array(X)).float()
    print(np.array(X).shape)
    deal_dataset = TensorDataset(gene_feature_tensor, torch.from_numpy(np.array(Y).astype(int)))
    # deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    # print(deal_dataset[:1])
    return deal_dataset, np.array(X).shape[0]

def load_multi_data(gene,idX,Y):
    X = []
    for cell in idX:
        cell_name = cell[0]
        value = gene.loc[gene['cell_name'] == replace_linetodot(cell_name)].values[:, 1:].tolist()[0]
        value = [math.log(x) if x > 0 else 0  for x in value]
        # value = [x if x > 0 else 0  for x in value]
        X.append(value)
    # print('gene', X[:1])
    gene_feature_tensor = torch.Tensor(np.array(X)).float()
    deal_dataset = TensorDataset(gene_feature_tensor, torch.from_numpy(np.array(Y).astype(int)))
    # deal_dataset = TensorDataset(torch.from_numpy(np.array(X).astype(float)), torch.from_numpy(np.array(Y).astype(int)))
    # print(deal_dataset[:1])
    return deal_dataset, np.array(X).shape[0]

def com_linearsize(linear_size,Con_layer,kernel_size):
    for i in range(Con_layer):
        linear_size = int(((linear_size + 2 * 1 - kernel_size) / 1 + 1) // 2)
    if Con_layer == 0:
        linear_size = 0
    return linear_size

def load_loader(train_dataset,test_dataset,test_size):
    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=32,
                              shuffle=False)

    test_loader = DataLoader(dataset=test_dataset,
                            batch_size=test_size,
                            shuffle=False)
    return train_loader, test_loader


def CNN_train(epoch, model, optimizer, train_loader, loss_fn, device):
    i = 0
    RLDCP_loader, gene_loader, multi_loader = train_loader
    for  (images_RLDCP, labels_RLDCP), (images_gene, labels_gene), (images_multi,labels_multi) in zip(RLDCP_loader, gene_loader,multi_loader):
        optimizer.zero_grad()
        labels = torch.Tensor(labels_RLDCP.type(torch.FloatTensor)).long()
        images_RLDCP = torch.unsqueeze(images_RLDCP.type(torch.FloatTensor), dim=1)  #
        images_gene = torch.unsqueeze(images_gene.type(torch.FloatTensor), dim=1)
        images_multi = torch.unsqueeze(images_multi.type(torch.FloatTensor), dim=1)
        images_gene = images_gene.to(device)
        images_RLDCP = images_RLDCP.to(device)
        images_multi = images_multi.to(device)
        labels = labels.to(device)
        outputs = model(images_RLDCP, images_gene,images_multi)
        train_loss = loss_fn(outputs, labels)
        train_loss.backward()
        optimizer.step()
        train_loss += train_loss.cpu().data * images_RLDCP.size(0)
        _, prediction = torch.max(outputs.data, 1)
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_RLDCP), len(RLDCP_loader.dataset),
                   100. * i / len(RLDCP_loader), train_loss.cpu().data ))
        # *images_RLDCP.size(0)
    return model, optimizer

def CNN_val(epoch, model, test_loader, loss_fn, test_size, device):
    RLDCP_loader, gene_loader, multi_loader = test_loader
    i = 0
    for  (images_RLDCP, labels_RLDCP), (images_gene, labels_gene),(images_multi,labels_multi) in zip(RLDCP_loader, gene_loader,multi_loader):
        images_RLDCP = torch.unsqueeze(images_RLDCP.type(torch.FloatTensor), dim=1)
        images_RLDCP = images_RLDCP.to(device)
        images_gene = torch.unsqueeze(images_gene.type(torch.FloatTensor), dim=1)
        images_gene = images_gene.to(device)
        images_multi = torch.unsqueeze(images_multi.type(torch.FloatTensor), dim=1)
        images_multi = images_multi.to(device)
        labels = torch.Tensor(labels_RLDCP.type(torch.FloatTensor)).long()
        labels = labels.to(device)
        outputs = model(images_RLDCP, images_gene,images_multi)
        test_result_matrix0 = outputs
        test_result_matrix = test_result_matrix0.cpu().detach().numpy()
        val_loss = loss_fn(outputs, labels)
        _, prediction = torch.max(outputs.data, 1)
        label_pred = prediction.cpu().numpy()
        label = labels.data.cpu().numpy()
        prediction_num = int(torch.sum(prediction == labels.data))
        val_accuracy = prediction_num / test_size
        print('Val Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
            epoch, i * len(images_RLDCP), len(RLDCP_loader.dataset),
                   100. * i / len(RLDCP_loader), val_loss.cpu().data ))
        i = i + 1
    return label_pred, label, val_loss, model, val_accuracy , test_result_matrix

class create_model(nn.Module):
    def __init__(self, model_para,transformer_para,eca_para,lstm_para):
        super(create_model, self).__init__()
        kernel_size, cnn_feature, dp, out_feature, Con_layer, linear_layer = model_para
        dp_trans, heads, blocks = transformer_para
        channel, b, eca_gamma = eca_para
        lstm_numlayer = lstm_para
        self.linear_layer = linear_layer
        linear_size_RLDCP_init = 2666
        linear_size_gene_init = 500  # 300
        self.Con_layer_RLDCP, self.Con_layer_gene  = Con_layer
        # linear_size_BOP = com_linearsize(linear_size_BOP_init, self.Con_layer_BOP, kernel_size)
        linear_size_RLDCP = com_linearsize(linear_size_RLDCP_init, self.Con_layer_RLDCP, kernel_size)
        linear_size_gene = com_linearsize(linear_size_gene_init,self.Con_layer_gene,kernel_size)
        self.linear_size_RLDCP = linear_size_RLDCP
        self.linear_size_gene = linear_size_gene
        self.cnn_feature = cnn_feature  # 通道数
        # 600 FNN的隐藏层emb
        self.transformer_encode_rna = fusion_transformer.TransformerEncoder(
                                                        linear_size_gene_init*2,linear_size_gene_init*2,linear_size_gene_init*2,
                                                        linear_size_gene_init*2,
                                                    [1, linear_size_gene_init*2],
                                                        linear_size_gene_init*2,linear_size_gene_init*2,
                                                    heads, blocks, dp_trans)
        # self.se_block = fusion_transformer.SE_block()
        self.eca_block = fusion_transformer.ECA_block_para(channel=channel, b = b, gamma=eca_gamma)
        # self.lstm =nn.LSTM(input_size=self.cnn_feature * self.linear_size_gene,
        #                    hidden_size=self.cnn_feature * self.linear_size_gene,
        #                    num_layers=3,bidirectional=True)
        self.lstm_rna =nn.LSTM(input_size=linear_size_gene_init,
                           hidden_size=linear_size_gene_init,
                           num_layers=lstm_numlayer,bidirectional=True)

        # LSTM+Transformer 和CNN出来之后的特征融合
        if self.Con_layer_RLDCP != 0:
            self.conv1_RLDCP = nn.Conv1d(in_channels=1, out_channels=cnn_feature, kernel_size=kernel_size, stride=1,
                                       padding=1)
            self.bn1_RLDCP = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_RLDCP = nn.ReLU()
            self.pool_RLDCP = nn.MaxPool1d(kernel_size=2)
            # self.dropout_RLDCP = nn.Dropout(dp)
            self.Con_RLDCP = nn.Sequential()
            for i in range(self.Con_layer_RLDCP - 1):
                layer_id = str(i + 2)
                self.Con_RLDCP.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_RLDCP.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_RLDCP.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_RLDCP.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                # self.Con_RLDCP.add_module("drop%s" % layer_id,nn.Dropout(dp))
        if self.Con_layer_gene != 0:
            self.conv1_gene = nn.Conv1d(in_channels=1, out_channels=cnn_feature, kernel_size=kernel_size, stride=1,
                                       padding=1)
            self.bn1_gene = nn.BatchNorm1d(num_features=cnn_feature)
            self.rule1_gene = nn.ReLU()
            self.pool_gene = nn.MaxPool1d(kernel_size=2)
            # self.dropout_RLDCP = nn.Dropout(dp)
            self.Con_gene = nn.Sequential()
            for i in range(self.Con_layer_gene - 1):
                layer_id = str(i + 2)
                self.Con_gene.add_module("conv%s" % layer_id,
                                        nn.Conv1d(in_channels=cnn_feature, out_channels=cnn_feature,
                                                  kernel_size=kernel_size, stride=1, padding=1))
                self.Con_gene.add_module("bach%s" % layer_id, nn.BatchNorm1d(num_features=cnn_feature))
                self.Con_gene.add_module("relu%s" % layer_id, nn.ReLU())
                self.Con_gene.add_module("maxp%s" % layer_id, nn.MaxPool1d(kernel_size=2))
                # self.Con_RLDCP.add_module("drop%s" % layer_id,nn.Dropout(dp))


        if linear_layer == 1:
            # LSTM 再 Transformer 前面 时使用
            self.linear_RLDCP = nn.Sequential()
            self.linear_RLDCP.add_module("linear1", nn.Linear(
                in_features=(cnn_feature * linear_size_RLDCP) + (cnn_feature * linear_size_gene ) +linear_size_gene_init*2,
                 out_features=(cnn_feature * linear_size_RLDCP + cnn_feature * linear_size_gene + linear_size_gene_init*2) // 2))
            self.linear_RLDCP.add_module("relu", nn.ReLU())
            self.linear_RLDCP.add_module("drop" ,nn.Dropout(dp))
            self.linear_RLDCP.add_module("linear2", nn.Linear(
                in_features=(((cnn_feature * linear_size_RLDCP) + (cnn_feature * linear_size_gene ) +linear_size_gene_init*2) // 2),
                                                               out_features = 4))


    def forward(self, x1,x2,x3):
        x_hic = torch.cat((x1,x3), dim = 2)
        valid_lens = None
        x2_lstm , _ = self.lstm_rna(x2) # 32，1，600
        x_rna = self.transformer_encode_rna(x2_lstm, valid_lens) # 32,1,600 # 经过LSTM+Transformer
        shape = x_rna.shape[2]
        if self.Con_layer_RLDCP != 0:
            x_hic = self.rule1_RLDCP(self.bn1_RLDCP(self.conv1_RLDCP(x_hic)))
            x_hic = self.pool_RLDCP(x_hic)
            x_hic= self.Con_RLDCP(x_hic)# 32 32 1331
            # x_hic_cnn = x_hic.view(-1, self.cnn_feature * self.linear_size_RLDCP) # 32,21216

        if self.Con_layer_gene != 0:
            x2 = self.rule1_gene(self.bn1_gene(self.conv1_gene(x2)))
            x2 = self.pool_gene(x2)
            x2 = self.Con_gene(x2)
            x2 = x2.view(-1, self.cnn_feature * self.linear_size_gene) # 经过CNN提取特征 32,2304
        # x_rna, _ = self.lstm(x_rna) # 出来是一个tuple
        x_hic_eca = self.eca_block(x_hic)
        x_hic_eca = x_hic_eca.view(-1, self.cnn_feature * self.linear_size_RLDCP)
        # x_hic_final = torch.cat((x_hic_eca,x_hic_cnn), dim=1) # ena和CNN提取的特征融合
        x_rna = x_rna.view(-1,  shape) # 展开
        x_rna_final = torch.cat((x2, x_rna),dim=1) # RNA：CNN和LSTM+Transformer融合
        x = torch.cat((x_hic_eca, x_rna_final), dim = 1) # hic和rna融合

        if self.linear_layer == 1:
            x = self.linear_RLDCP(x)

        x = nn.functional.log_softmax(x, dim=1)
        return x



def CNN_1D_network(RLDCP,gene_feature,multi_feature, tr_x, tr_y, X_test, y_test, lr, model_para, alpha, gamma,transformer_para,eca_para,lstm_para):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    fold = 'test'
    seed_torch()
    train_dataset_RLDCP, train_size_RLDCP = load_rscip_data(RLDCP, tr_x, tr_y)
    test_dataset_RLDCP, test_size_RLDCP = load_rscip_data(RLDCP, X_test, y_test)
    train_loader_RLDCP, test_loader_RLDCP = load_loader(train_dataset_RLDCP,test_dataset_RLDCP, test_size_RLDCP)
    train_dataset_gene, train_size_gene = load_gene_data(gene_feature, tr_x, tr_y)
    test_dataset_gene, test_size_gene = load_gene_data(gene_feature, X_test, y_test)
    train_loader_gene, test_loader_gene = load_loader(train_dataset_gene,test_dataset_gene, test_size_gene)
    train_dataset_multi_feature, train_size_multi_feature = load_multi_data(multi_feature, tr_x, tr_y)
    test_dataset_multi_feature, test_size_multi_feature = load_multi_data(multi_feature, X_test, y_test)
    train_loader_multi_feature, test_loader_multi_feature = load_loader(train_dataset_multi_feature,test_dataset_multi_feature,
                                                                        test_size_multi_feature)
    model = create_model(model_para,transformer_para,eca_para,lstm_para)
    device = try_gpu(1)
    model.to(device)
    print(model)
    train_loader = [train_loader_RLDCP, train_loader_gene,train_loader_multi_feature]
    test_loader = [test_loader_RLDCP, test_loader_gene,test_loader_multi_feature]
    num_epochs = 40
    min_loss = 100000.0
    optimizer = Adam(model.parameters(), lr=lr)
    # loss_fn = nn.CrossEntropyLoss()
    loss_fn = MultiClassFocalLossWithAlpha(alpha=alpha, gamma=gamma)
    # loss_fn = nn.CrossEntropyLoss()
    for epoch in range(num_epochs):
        model.train()
        model, optimizer = CNN_train(epoch, model, optimizer, train_loader, loss_fn,device)
        model.eval()
        test_label, label, test_loss, model, test_accuracy, test_result_matrix = CNN_val(epoch, model, test_loader,
                                                                          loss_fn, test_size_RLDCP,device)
    torch.cuda.empty_cache()

    return test_accuracy, test_label, label, test_result_matrix