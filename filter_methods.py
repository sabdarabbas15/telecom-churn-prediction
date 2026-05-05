'''
THIS FILE CONTAINS FEATURE SELECTION METHODS
'''
#AVOIDING WARNINGS
import warnings
warnings.filterwarnings('ignore')

#IMPORTING LIBRARYS
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.feature_selection import VarianceThreshold
from scipy.stats import pearsonr

#IMPORTING FILES(IMPUTATION)
import logging_code
from logging_code import setup_logging
logger = setup_logging('filter_methods')

def fm(x_train_num, x_test_num, y_train, y_test): #CREATING FUNCTION
    try: #TRY BLOCK
        logger.info(f'Before x_train_num columns: {x_train_num.shape}:\n: {x_train_num.columns}')#CHECKING
        logger.info(f'Before x_test_num columns: {x_test_num.shape}:\n: {x_test_num.columns}') #CHECKING
        reg = VarianceThreshold(threshold=0.01)  #MODEL
        reg.fit(x_train_num)   #FITTING X_TRAIN_NUM COLUMNS IN THE MODEL
        logger.info(f'good columns: {sum(reg.get_support())} : {x_train_num.columns[reg.get_support()]}') #CHECKING
        logger.info(f'bad columns: {sum(~reg.get_support())} : {x_train_num.columns[~reg.get_support()]}') #CHECKING
        x_train_num = x_train_num.drop(['SeniorCitizen_trim'],axis=1)  #DROPPING BAD COLUMN IN X_TRAIN
        x_test_num = x_test_num.drop(['SeniorCitizen_trim'], axis=1)   #DROPPING BAD COLUMN IN X_TEST
        logger.info(f'After x_train_num columns: {x_train_num.shape}:\n: {x_train_num.columns}') #VERIFYING
        logger.info(f'After x_test_num columns: {x_test_num.shape}:\n: {x_test_num.columns}')  #VERIFYING

        logger.info(f'--------------------------------hypothesis testing part------------------------------------------')

        c = []   #EMPTY VARIABLE
        for i in x_train_num.columns:   #FOR LOOP
            r = pearsonr(x_train_num[i], y_train)  #STORING DATA INNTO NEW VARIABLE
            c.append(r)    #STORING DATA INTO EMPTY COLUMN
        t = np.array(c)     #NEW VARIABLE
        p_value = pd.Series(t[:, 1], index=x_train_num.columns)  #P_VALUE
        # p = 0
        # f = []
        # for i in p_value:
        #     if i < 0.05:
        #         f.append(x_train_num.columns[p])
        #     p = p + 1
        # print(x_train_num.columns)
        # print(f) # good columns
        logger.info(f'After HYPOTHESIS TESTING train columns : {x_train_num.shape} \n :  {x_train_num.columns}')#VERIFYING
        logger.info(f'After HYPOTHESIS TESTING test columns : {x_test_num.shape} \n : {x_test_num.columns}')#VERIFYING

        return x_train_num,x_test_num #RETURNING
    except Exception as e:  #EXCEPTIONAL(ERROR HANDLING METHOD)
        err_type,err_line,err_msg = sys.exc_info()
        logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")