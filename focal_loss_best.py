import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
# https://zhuanlan.zhihu.com/p/562641889
# 首先，明确一下loss函数的输入：
# 一个pred，shape为 (bs, num_classes)，并且未经过softmax ；
# 一个target，shape为 (bs)，也就是一个向量，并且未经过one_hot编码。
# 通过前面的公式可以得出，我们需要在loss实现是做三件事情：
#
# 找到当前batch内每个样本对应的类别标签，然后根据预先设置好的alpha值给每个样本分配类别权重
# 计算当前batch内每个样本在类别标签位置的softmax值，作为公式里的
#  ，因为不管是focal loss还是cross_entropy_loss，每个样本的n个概率中不属于真实类别的都是用不到的
# 计算原始的cross_entropy_loss ，但不能求平均，需要得到每个样本的cross_entropy_loss ，因为需要对每个样本施加不同的权重

def try_gpu(i=0):  #@save
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')

class MultiClassFocalLossWithAlpha(nn.Module):
    def __init__(self, alpha=[], gamma=2, reduction='mean'):
        """
        :param alpha: 权重系数列表，四分类中每一类的权重取得都是该类别出现频率的倒数
        :param gamma: 困难样本挖掘的gamma,gamma是给分错的样本的权重调大。
        :param reduction:
        """
        super(MultiClassFocalLossWithAlpha, self).__init__()
        device = try_gpu(1)
        self.alpha = torch.tensor(alpha).to(device)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 4)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss

class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30):
        super(LDAMLoss, self).__init__()
        device = try_gpu(0)
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.cuda.FloatTensor(m_list)
        self.m_list = m_list.to(device)
        assert s > 0
        self.s = s
        self.weight = weight

    def forward(self, x, target):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.cuda.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        return F.cross_entropy(self.s * output, target, weight=self.weight)

def focal_loss(input_values, alpha,gamma):
    """Computes the focal loss"""
    p = torch.exp(-input_values)
    loss = alpha * (1 - p) ** gamma * input_values
    return loss.mean()

class FocalLoss(nn.Module):
    def __init__(self, weight, alpha=[], gamma=1):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        device = try_gpu(0)
        self.gamma = gamma
        self.alpha = torch.tensor(alpha).to(device)
        self.weight = weight.to(device)

    def forward(self, input, target):
        return focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), self.alpha[target],self.gamma)