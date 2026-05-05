'''
THIS FILE CONTAINS VARIABLE TRANSFORMATION AND HANDLING OUTLIERS
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
from scipy.stats import yeojohnson

#IMPORTING FILES(IMPUTATION)
import logging_code
from logging_code import setup_logging
logger = setup_logging('var_out')


def vt_outliers(x_train_num, x_test_num):
    try:
        logger.info(f'Before x_Train columns names : {x_train_num.columns}')   #CHECKING
        logger.info(f'Before x_test columns names : {x_test_num.columns}')    #CHECKING
        for i in x_train_num.columns:       #FOR LOOP
            x_train_num[i+'_yeo'],lam_val = yeojohnson(x_train_num[i])   #CREATING NEW COLUMN IN X_TRAIN
            x_test_num[i+'_yeo'],lam_val = yeojohnson(x_test_num[i])   #CREATING NEW COLUMN IN X_TEST
            x_train_num = x_train_num.drop([i],axis=1)                #DROPPING OLD COLUMN IN X_TRAIN
            x_test_num = x_test_num.drop([i],axis=1)                 #DROPPING OLD COLUMN IN X_TEST
            #TRIMMING
            iqr = x_train_num[i+'_yeo'].quantile(0.75) - x_train_num[i+'_yeo'].quantile(0.25)  #FINDING IQR(Q3-31)
            upper_limit = x_train_num[i+'_yeo'].quantile(0.75) + (1.5 *iqr)  #FINDING UP_L(Q3+1.5*IQR)
            lower_limit = x_train_num[i+'_yeo'].quantile(0.25) - (1.5 * iqr) #FINDING L_L(Q1-1.5*IQR)
            x_train_num[i+'_trim'] = np.where(x_train_num[i+'_yeo'] > upper_limit, upper_limit,
                                              np.where(x_train_num[i+'_yeo'] < lower_limit,lower_limit,
                                                       x_train_num[i+'_yeo'])) #CREATING NEW IN X_TRAIN
            x_test_num[i +'_trim'] = np.where(x_test_num[i+'_yeo'] > upper_limit, upper_limit,
                                                np.where(x_test_num[i+'_yeo'] < lower_limit, lower_limit,
                                                         x_test_num[i+'_yeo'])) #CREATING NEW IN X_TEST
            x_train_num = x_train_num.drop([i+'_yeo'],axis=1)    #DROPPING OLD COLUMN IN X_TRAIN
            x_test_num = x_test_num.drop([i+'_yeo'],axis=1)     #DROPPING OLD COLUMN IN X_TEST
            logger.info(f'After x_Train columns names : {x_train_num.columns}') #VERIFYING
            logger.info(f'After x_test columns names : {x_test_num.columns}')   #VERIFYING

            return x_train_num,x_test_num
    except Exception as e:  #EXCEPTIONAL(ERROR HANDLING METHOD)
        err_type,err_line,err_msg = sys.exc_info()
        logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")