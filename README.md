# MomicPred

MomicPred: A cell cycle prediction framework based on dual-branch multi-modal feature fusion for single-cell multi-omics data

## Framework
![image](hic-rna架构-1.jpg)

 a Multi-modal feature extraction. MomicPred extracts distinct features from multi-omics data. Specifically, we extract three feature sets: random TAD-like contact probability (RLDCP) feature set, high expression gene (HEG) feature set, and cell-level macroscopic information (CLMI) feature set. b Dual-branch multi-modal feature fusion model. We develop a dual-branch multi-modal feature fusion model to deeply explore complementary information among features, thereby enhancing the accuracy and robustness of cell cycle prediction. c The usage of MomicPred. Directly apply the trained model to predict cell cycles based on multi-omics data.

## Overview

The folder "4DN" contains of the preocessing of 4DN sci-HiC dataset.  
The folder "Flyamer" contains of the preocessing of Flyamer et al. dataset.  
The folder "Ramani" contains of the preocessing of Ramani et al. dataset.  
The folder "Lee" contains of the preocessing of Lee et al. dataset.  
The folder "Collombet" contains of the preocessing of Collombet et al. dataset.  
The folder "Nagano" contains of the preocessing of Nagano et al. dataset.  
The folder "Data_filter" contains processed data from six datasets.  

In the above six folders, each folder contains folder "generate_features" and folder "model":  
The folder "generate_features" caontains of the peocess of extracting multi-omic features.  
The folder "model" contains the framework of models.  

In folder "generate_features" of the above six folders:  
The file "generate_feature_sicp.py" is the code to extract small intra-domain contact probability (SICP) feature set.  
The file "generate_feature_ssicp.py" is the code to extract smoothed small intra-domain contact probability (SSICP) feature set.  
The file "generate_feature_sbcp.py" is the code to extract smoothed bin contact probabilit (SBCP) feature set.  

In folder "model" of the above six folders:  
The file "final_sicp_ssicp_sbcp.py" is the code of model.  
The file "focal_loss,py" is the code of loss function.  

## Dependency
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
