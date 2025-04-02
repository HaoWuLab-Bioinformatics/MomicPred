# MomicPred

MomicPred: A cell cycle prediction framework based on dual-branch multi-modal feature fusion for single-cell multi-omics data

## Framework
![image](framework.jpg)

(**a**) Multi-modal feature extraction. MomicPred extracts distinct features from multi-omics data. Specifically, we extract three feature sets: random TAD-like contact probability (RLDCP) feature set, high expression gene (HEG) feature set, and cell-level macroscopic information (CLMI) feature set. 

(**b**) Dual-branch multi-modal feature fusion model. We develop a dual-branch multi-modal feature fusion model to deeply explore complementary information among features, thereby enhancing the accuracy and robustness of cell cycle prediction. 

(**c**) The usage of MomicPred. Directly apply the trained model to predict cell cycles based on multi-omics data.


## Overview

1. HEG_model : This folder holds the construction of a predictive model using the HEG feature set alone.

2. RLDCP_model : This folder holds the construction of a predictive model using the RLDCP feature set alone.

3. best_para : It is a fusion prediction model based on RLDCP, CLMI, and HEG feature sets.

4. Dataset : This folder holds feature sets generated in the hic_feature and rna_feature folders.

5. Data_Preparation :  This folder holds the code for Data_preparation.

6. RLDCP+HEG_model : This folder holds the construction of a predictive model using the RLDCP and HEG feature sets.

7. only_hic_model : This folder holds the construction of a predictive model using the RLDCP and CLMI feature sets.

8. hic_feature :  This folder holds the code about the extraction of RLDCP and CLMI.

9. rna_feature :  This folder holds the code about the extraction of HEG.

## Dependency
Mainly used libraries:   
Python 3.9    
pytorch  
xlsxwriter  
sklearn  
numpy  
collection  
scipy

## Usage
Perform the following steps in each folder named after the dataset:  
First, you should extract features of data, you can run the script to extract random TAD-like contact probability (RLDCP) feature set and high expression gene (HEG) feature set  as follows:  
`python ./hic_feature/extract_hic_features.py`  
`python ./rna_feature/extract_TopGene.py`  
Then run the script as follows to compile and run MomicPred:  
`python best_para.py`     
