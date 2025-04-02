

import numpy as np



    # 用于重启随机游走平滑
    # 参数rp是重启概率
def random_walk_imp(matrix, rp):
    # row获取了接触矩阵的行数
    row, _ = matrix.shape
    row_sum = np.sum(matrix, axis=1)
    # row_sum.shape[0]可以求出row_sum这个列表中有多少个元素。
    for i in range(row_sum.shape[0]):
        # 因为在下面divide函数中arr2作为分母，不能为0，所以要把所有=0的赋值为0.001。
        if row_sum[i] == 0:
            row_sum[i] = 0.001
    nor_matrix = np.divide(matrix.T, row_sum).T
    Q = np.eye(row)
    I = np.eye(row)
    for i in range(30):
        Q_new = (1 - rp) * np.dot(Q, nor_matrix) + rp * I
        delta = np.linalg.norm(Q - Q_new)
        Q = Q_new.copy()
        if delta < 1e-6:
            break
    return Q

def sum_matrix(matrix):
    # 计算染色质的总接触
    U = np.triu(matrix, 1)
    # 返回矩阵的上三角部分，不包括对角线
    D = np.diag(np.diag(matrix)) 
    return sum(sum(U + D))  

def Small_Domain_Struct_Contact_pro(contact_matrix, max_length, scale):
    random_matrix = random_walk_imp(contact_matrix, rp=0.1)  # 经过重启随机游走处理
    contact_matrix = np.array(random_matrix) # 染色体的接触矩阵
    new_matrix = np.zeros((max_length+2*scale,max_length+2*scale)) # 扩增接触矩阵
    RLDCP = []
    chr_total = sum_matrix(contact_matrix) # 所有接触数的总和
    for i in range(max_length):
        for j in range(max_length):
            new_matrix[i+scale,j+scale] = contact_matrix[i,j]  # 这是把接触信息放进去
    for i in range(max_length):
        bin = i+scale
        a  = sum_matrix(new_matrix[bin-scale:bin+scale+1,bin-scale:bin+scale+1])  
        if a == 0:
            RLDCP.append(float(0))
        else:
            RLDCP.append(float(a/chr_total))
    return RLDCP



def con_ran(cell_name, chr_name, max_length):  # 因为TAD域大小不一样，所以scale可以采用1，3，5拼接的方式
    # con_ran(cell_id, type, chr_name, max_len, cell_dict)
    print(cell_name)
    file_path = "../GSE_final_bin_count/%s/%s.txt" % (cell_name, chr_name)
    chr_file = open(file_path)
    scale = 1
    lines = chr_file.readlines()
    contact_matrix = np.zeros((max_length, max_length))
    no_zero_num = 0
    for line in lines:
        # bin1，bin2是两个染色体片段的编号，num是bin1和bin2的接触数
        bin1, bin2, num = line.split()
        contact_matrix[int(bin1), int(bin2)] += int(float(num))
        if num != 0:
            no_zero_num += 1
        if bin1 != bin2:
            contact_matrix[int(bin2), int(bin1)] += int(float(num))
    print(no_zero_num)
    # no_zero_num = 0
    # for row in range(contact_matrix.shape[0]):
    #     for col in range(contact_matrix.shape[1]):
    #         if contact_matrix[row][col]!=0:
    #             no_zero_num+=1
    flag = 0

    for row in range(contact_matrix.shape[0]):
        for col in range(contact_matrix.shape[1]):
            if contact_matrix[row][col] != 0:
                if flag <= no_zero_num * 0.45:
                    if np.random.rand() > 0.5:
                        contact_matrix[row][col] += 1
                        flag += 1
                    else:
                        contact_matrix[row][col] -= 1
                        flag += 1
    print('flag:', flag)

    RLDCP = Small_Domain_Struct_Contact_pro(contact_matrix, max_length, scale)

    return RLDCP

def main():

    file_list = np.loadtxt('../cell_cycle.txt', dtype=str)  #
    need_cell = []  # 选出需要的细胞
    for file in file_list:
        need_cell.append(file[0])
    resolutions = [1000000]
    # tps = sorted(types)
    # print(tps) ['1CSE', '2CSE', '4CSE', '64CSE', '8CSE']
    f = open('mm10.main.nochrM.chrom.sizes')
    index = {}
    for resolution in resolutions:
        index[resolution] = {}
        lines = f.readlines()
        for line in lines:
            chr_name, length = line.split()
            # 经过下面这行代码以后，chr_name的值由human_chr1变为chr1
            # chr_name = chr_name.split('_')[1]
            # max_len+1是指一个长度为length的染色体在resolution分辨率下能分成max_len+1块
            # 为什么要+1？因为int(10/3)=3，是向下取整的，多出来的那一截，也要做一块。
            max_len = int(int(length) / resolution)
            # index二维字典中index[resolution][chr_name]存的是染色体chr_name在分别率为resolution时能分出的块数
            index[resolution][chr_name] = max_len + 1
        f.seek(0, 0)
    # 关闭文件流
    f.close()
    print(index)
    # 1mbp就是指，1M（1000000）个碱基对作为分辨率，所以rindex就是用来标注index的分辨率的
    rindex = "1mbp"
    # p = Pool(20) # 小鼠需要弄到20
    cell_dict = {}
    cell_number = 0
    chr_list = sorted(index[resolution].keys())
    for cell in need_cell[1:]:
        cell_dict[cell] = {}
        for chr in chr_list:
            # c_num在循环中是0-43（因为range（44）是0-43）、0-213、0-257、0-109

            max_len = index[resolution][chr]
            # print(index[resolution])
            # args = [[rindex, cell_id, type, chr_name, index[resolution][chr_name]] for chr_name in
            #         index[resolution]]

            # print(args)
            RLDCP = con_ran(cell, chr, max_len)
            cell_dict[cell][chr] = RLDCP
    out_path = 'RLDCP_1.npy'  # 数据扰动率为15%
    # 1:0.67   3: 0.607
    np.save(out_path, cell_dict)


if __name__ == '__main__':
    main()
    # 特征思路：  (1):当前接触数在指定TAD域中的概率 (2) bin在对角线上的概率  （3）：线性邻居+空间邻居
    # 扰动实验 Drop实验
    # 细胞嵌入方法 KPCA t-sne 证明我们特征的有效性
