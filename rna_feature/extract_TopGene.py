

import numpy as np
import pandas as pd
import scanpy as sc

# 按间距中的绿色按钮以运行脚本。
if __name__ == '__main__':

    rna = sc.read_csv('./outfile.tsv', delimiter='\t')
    print(rna)
    file_list = np.loadtxt('cell_cycle.txt', dtype=str)   #
    need_cell = []  # 选出需要的细胞
    for file in file_list:
        need_cell.append(file[0])
    # # print(df) # AnnData object with n_obs × n_vars = 50463 × 7469  50463个基因
    sc.pp.highly_variable_genes(rna, n_top_genes=300, flavor="cell_ranger",inplace=True)
    rna_gene = rna.var # 50463,4 把为True的基因选出来
    rna_value = rna.X # 还是原先的值 7469 * 50463
    col_name = rna.var_names.tolist() # 列为基因
    row_name = rna.obs_names.tolist() # 行为细胞
    rna_gene_df = pd.DataFrame(rna_value, index=row_name, columns=col_name)
    top_gene_name = rna_gene.index[rna_gene['highly_variable'] == True].to_list()
    rna_gene_filtered = rna_gene_df[rna_gene_df.index.isin(need_cell)]
    rna_final = rna_gene_filtered[rna_gene_filtered.columns[rna_gene_filtered.columns.isin(top_gene_name)]]

    print(rna_final.shape)
    rna_final.to_csv('./Dataset/TopGene_50.txt',sep='\t', index=True)

