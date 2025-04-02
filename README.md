# MomicPred

MomicPred: A cell cycle prediction framework based on dual-branch multi-modal feature fusion for single-cell multi-omics data

## Framework
![image](framework.jpg)

(**a**) Multi-modal feature extraction. MomicPred extracts distinct features from multi-omics data. Specifically, we extract three feature sets: random TAD-like contact probability (RLDCP) feature set, high expression gene (HEG) feature set, and cell-level macroscopic information (CLMI) feature set. 

(**b**) Dual-branch multi-modal feature fusion model. We develop a dual-branch multi-modal feature fusion model to deeply explore complementary information among features, thereby enhancing the accuracy and robustness of cell cycle prediction. 

(**c**) The usage of MomicPred. Directly apply the trained model to predict cell cycles based on multi-omics data.


## Overview

1. BCP : This folder holds the construction of a predictive model using the BCP feature set alone.

2. CDD : This folder holds the construction of a predictive model using the CDD feature set alone.

3. Construct_fusion_model : This folder holds the construction of a fusion prediction model for the three feature sets.

4. Data : This folder holds the CDD, BCP and SICP feature sets generated in the Feature_sets_extraction phase.

5. Data_Preparation :  This folder holds the code for Data_preparation.

6. Feature_sets_extraction :  This folder holds the code to perform feature-set extraction.

7. Raw_Data : This folder contains the raw data used in this study.

8. SICP : This folder holds the construction of a predictive model using the SICP feature set alone.

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
First, you should extract features of data, you can run the script to extract smoothed bin contact probabilit (SBCP), small intra-domain contact probability (SICP), and smoothed small intra-domain contact probability (SSICP) features as follows:  
`python ./generate_features/generate_feature_sicp.py`  
`python ./generate_features/generate_feature_ssicp.py`  
`python ./generate_features/generate_feature_sbcp.py`    
Then run the script as follows to compile and run CTPredictor:  
`python ./model/final_sicp_ssicp_sbcp.py`     
