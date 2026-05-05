'''
THIS FILE CONTAINS NULL VALUES HANDLING TECHNIQUE
'''
#IMPORTING LIBRARIES
import os
import sys
import numpy as np
import pandas as pd
import sklearn

#AVOIDING WARNINGS
import warnings
warnings.filterwarnings('ignore')

#IMPORTING FILES(IMPUTATION)
import logging_code
from logging_code import setup_logging
logger = setup_logging('mode_hmv')

#CREATING DEF FUNCTION
def handle_missing_values(x_train,x_test):
    try:                 #TRYBLOCK
        logger.info(f'Before handling missing values in x_train: {x_train.shape} \n : {x_train.columns}: {x_train.isnull().sum()}')#CHECKING NULL VALUES
        logger.info(f'After handling missing values in x_test: {x_train.shape} \n : {x_train.columns} : {x_test.isnull().sum()}')#CHECKING NULL VALUES
        for i in x_train.columns:  # FORLOOP
              if x_train[i].isnull().sum() > 0:
                  x_train[i +'_mode'] = x_train[i].fillna(x_train[i].mode()[0])
                  x_test[i +'_mode'] = x_test[i].fillna(x_test[i].mode()[0])
                  x_train = x_train.drop([i], axis=1)
                  x_test = x_test.drop([i], axis=1)

        logger.info(f'After handling missing values in x_train: {x_train.isnull().sum()}')#CHECKINBG NULL VALUES
        logger.info(f'After handling missing values in x_test: {x_test.isnull().sum()}')#CHECKINBG NULL VALUES

        return x_train,x_test    #RETURNING
    except Exception as e:  #EXCEPTIONAL(ERROR HANDLING METHOD)
        err_type,err_line,err_msg = sys.exc_info()
        logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")
