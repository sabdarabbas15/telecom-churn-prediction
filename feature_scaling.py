'''
THIS FILE CONTAINS FEATURE SCALING PART
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
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler #z-score
import pickle

#IMPORTING FILES(IMPUTATION)
import logging_code
from logging_code import setup_logging
logger = setup_logging('feature_scaling')
from all_models import common
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,classification_report,confusion_matrix


def fs(x_train, y_train, x_test, y_test):
    try:
        logger.info(f'Training data independent size : {x_train.shape}')
        logger.info(f'Training data dependent size : {y_train.shape}')
        logger.info(f'testing data independent size : {x_test.shape}')
        logger.info(f'testing data dependent size : {y_test.shape}')
        logger.info(f'before {x_train.head(1)}')
        sc=StandardScaler()
        sc.fit(x_train)
        x_train_sc = sc.transform(x_train)
        x_test_sc = sc.transform(x_test)

        with open('standard_scaler.pkl','wb') as f:
            pickle.dump(sc,f)

        #common(x_train_sc,y_train,x_test_sc,y_test)
        reg=XGBClassifier()
        reg.fit(x_train_sc, y_train) #TRANING COMPLETED
        logger.info(f'test accuracy: {accuracy_score(y_test,reg.predict(x_test_sc))}')
        logger.info(f'test confusion matrix: {confusion_matrix(y_test,reg.predict(x_test_sc))}')
        logger.info(f'classificatio report: {classification_report(y_test,reg.predict(x_test_sc))}')

        with open('Model.pkl','wb') as t:
            pickle.dump(reg,t)

    except Exception as e:  #EXCEPTIONAL(ERROR HANDLING METHOD)
        err_type,err_line,err_msg = sys.exc_info()
        logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")