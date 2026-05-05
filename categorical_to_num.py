'''
THIS FILE CONTAINS DATA WHICH IS CHNAGING CATEGORICAL TO NUMERICAL
'''
#AVOIDING WARNINGS
import warnings

from fontTools.merge.util import first

warnings.filterwarnings('ignore')

#IMPORTING LIBRARYS
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#IMPORTING FILES(IMPUTATION)
import logging_code
from logging_code import setup_logging
logger = setup_logging('categorical_to_num')
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder


def c_t_n(x_train_cat, x_test_cat):  #CREATING FUNCTION
    try:  #TRY BLOCK
        logger.info(f'===========================nominalEncoder=============================================================')
        logger.info(f"Before x_train_cat: {x_train_cat.shape}:\n:{x_train_cat.columns}")  #CHECKING
        logger.info(f"Before x_test_cat: {x_test_cat.shape}:\n:{x_test_cat.columns}") #CHECKING
        oh = OneHotEncoder(drop='first')  #NOMINALENCODER DROPING FIRST
        oh.fit(x_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                            'InternetService', 'OnlineSecurity', 'OnlineBackup',
                            'DeviceProtection',
                            'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling',
                            'PaymentMethod', 'Telecom Brands']])  #FITTING NOMINAL COLUMNS
        values_train = oh.transform(x_train_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                  'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                                 'DeviceProtection',
                                                  'TechSupport', 'StreamingTV', 'StreamingMovies','PaperlessBilling',
                                                 'PaymentMethod', 'Telecom Brands']]).toarray() # TRANFORMING NOMINAL COLUMNS IN X_TRAIN
        values_test = oh.transform(x_test_cat[['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                  'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                                  'DeviceProtection',
                                                  'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                                                  'PaymentMethod', 'Telecom Brands']]).toarray() #TRANSFORMING NOMINAL COLUMNS IN X_TEST
        t1 = pd.DataFrame(values_train) #ADDING DATA INTO NEW EXCEL SHEET
        t2 = pd.DataFrame(values_test)  #ADDING DATA INTO NEW EXCEL SHEET
        t1.columns = oh.get_feature_names_out()  #GETTING NAMES
        t2.columns = oh.get_feature_names_out()  #GETTING NAMES
        x_train_cat.reset_index(drop=True, inplace=True) #RESETING INDEX
        x_test_cat.reset_index(drop=True, inplace=True) #RESETING INDEX
        t1.reset_index(drop=True, inplace=True)   #RESETING INDEX
        t2.reset_index(drop=True, inplace=True)  #RESETING INDEX
        x_train_cat = pd.concat([x_train_cat, t1], axis=1)  #CONCATNING
        x_test_cat = pd.concat([x_test_cat, t2], axis=1) #CONCATNING
        x_train_cat = x_train_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                  'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                                  'DeviceProtection',
                                                  'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                                                  'PaymentMethod', 'Telecom Brands'], axis=1) #DROPPING COLUMNS
        x_test_cat = x_test_cat.drop(['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                                                  'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                                  'DeviceProtection',
                                                  'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling',
                                                  'PaymentMethod', 'Telecom Brands'], axis=1) #DROPPING COLUMNS
        logger.info(f'After NOMINAL x_train_cat Columns {x_train_cat.shape} \n : {x_train_cat.columns}') #VERIFYING
        logger.info(f'After NOMINAL x_test_cat Columns {x_test_cat.shape} \n : {x_test_cat.columns}') #VERIFYING

        logger.info(f'===========================OrdinalEncoder=============================================================')

        od = OrdinalEncoder()  #ODINAL ENCODER
        od.fit(x_train_cat[['Contract']])  #FITTING ODINAL COLUMN
        results_train = od.transform(x_train_cat[['Contract']]) #TRANSFORMING ODINAL COLOUMNS OF X_TRAIN
        results_test = od.transform(x_test_cat[['Contract']]) #TRANSFORMING ODINAL COLOUMNS OF X_TEST
        p1 = pd.DataFrame(results_train) #ADDING DATA INTO NEW EXCEL SHEET
        p2 = pd.DataFrame(results_test) #ADDING DATA INTO NEW EXCEL SHEET
        p1.columns = od.get_feature_names_out()+'_od' #GETTING NAMES+OD ADDING
        p2.columns = od.get_feature_names_out()+'_od' #GETTING NAMES+OD ADDING
        p1.reset_index(drop=True, inplace=True)  #RESETING INDEX
        p2.reset_index(drop=True, inplace=True)  #RESETING INDEX
        x_train_cat = pd.concat([x_train_cat, p1],axis=1)  #CONCATNING
        x_test_cat = pd.concat([x_test_cat, p2],axis=1)   #CONCATNING
        x_train_cat = x_train_cat.drop(['Contract'], axis=1) #DROPPING COLUMNS
        x_test_cat = x_test_cat.drop(['Contract'], axis=1) #DROPPING COLUMNS
        logger.info(f'After ODINAL x_train_cat Columns {x_train_cat.shape} \n : {x_train_cat.columns}') #VERIFYING
        logger.info(f'After ODINAL x_test_cat Columns {x_test_cat.shape} \n : {x_test_cat.columns}') #VERIFYING

        logger.info(f'Train NULL VALUES : \n{x_train_cat.isnull().sum()}') #VERIFYING
        logger.info(f'Test NULL VALUES : \n{x_test_cat.isnull().sum()}') #VERIFYING


        return  x_train_cat,x_test_cat  #RETURNING

    except Exception as e:  #EXCEPTIONAL(ERROR HANDLING METHOD)
        err_type,err_line,err_msg = sys.exc_info()
        logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")