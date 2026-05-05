'''
THIS FILE CONTAINS DATA CLEANING AND MODEL DEVELOPMENT PART
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
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import pickle

#IMPORTING FILES(IMPUTATION)
import logging_code
from logging_code import setup_logging
logger = setup_logging('main')
from mode_hmv import handle_missing_values
from var_out import vt_outliers
from filter_methods import fm
from categorical_to_num import c_t_n
from feature_scaling import fs



#CREATING CLASS(CONSTRUCTOR)
class CHURN:
    def __init__(self,path):                   #CREATING DEF(FUNCTION)
        try:        #WRITING CODE UNDER 'TRY' BLOCK
            logger.info(f'---------------------------------^cunstructor part^------------------------------------------')
            self.path = path
            self.df = pd.read_csv(self.path)       #UPLOADING DATA INTO VARIABLE CALLED 'DF'
            logger.info(f'TOTAL DATA SIZE: {self.df.shape}')    #FINDING DATA SHAPE(SIZE)
            logger.info(f"TOTAL INFORMATION ABOUT DATA: {self.df.info()}")   #CHECKING
            self.df['TotalCharges'] = self.df['TotalCharges'].replace(' ',np.nan) #REPLACING
            self.df['TotalCharges'] = pd.to_numeric(self.df['TotalCharges']) #CHANGING DTYPE
            logger.info(f"TOTAL NULL VALUES: \n :{self.df.isnull().sum()}")  #CHECKING
            self.df['Telecom Brands'] = self.df['PaymentMethod'].replace(
                {'Electronic check':'JIO', 'Mailed check':'AIRTEL', 'Bank transfer (automatic)':'BSNL',
                 'Credit card (automatic)':'IDEA'})     #CREATING A NEW COLUMN CALLED 'TELECOM BRANDS'
            logger.info(f'Successfully created Telecom Brands column')  #VERIFYING
            logger.info(f'After created new column now the Total data size: {self.df.shape}') #CHECKING DATA SIZE
            self.df = self.df.drop(['customerID'],axis=1)  #DROPING CUSTOMERID COLUMN
            logger.info(f'Successfully droped customer ID column') #VERIFYING DROP OR NOT
            self.x = self.df.drop(['Churn'],axis=1) #'INDEPENDENT' COLUMN
            self.y = self.df['Churn'] #DEPENDENT COLUMN
            self.x_train,self.x_test,self.y_train,self.y_test = train_test_split(self.x,self.y,test_size=0.2,random_state=42) #HERE GIVING X_TRAIN,X_TEST,Y_TRAIN,Y_TEST TO TRAIN TEST SPLIT
            self.y_train = self.y_train.map({'Yes': 1, 'No': 0})
            self.y_test = self.y_test.map({'Yes': 1, 'No': 0})
            logger.info(f'Successfully changed yes:1 & no:0 in y_test&y_train') #VERIFYING
            self.y_train = self.y_train.astype(int)
            self.y_test = self.y_test.astype(int)
            logger.info(f'---------------------------------^cunstructor part^------------------------------------------')
        except Exception as e:                       #EXCEPTIONAL(ERROR HANDLING METHOD)
            err_type,err_line,err_msg =sys.exc_info()
            logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")

    def missing_values(self): #CREATING FUNCTION
        try:          #TRYBLOCK
            logger.info(f'---------------------------------^missing values part^---------------------------------------')
            logger.info(f'Before handling missing values in x_train:{self.x_train.columns}:\n:{self.x_train.shape}:\n:{self.x_train.isnull().sum()}')#CHECKING
            logger.info(f'Before handling missing values in x_test:{self.x_test.columns}:\n:{self.x_test.shape}:\n:{self.x_test.isnull().sum()}')#CHECKING
            self.x_train,self.x_test = handle_missing_values(self.x_train,self.x_test) #GETTING DATA OR EXCUTING DATA
            logger.info(f'After handling missing values in x_train:{self.x_train.columns}:\n:{self.x_train.shape}:\n:{self.x_train.isnull().sum()}')#VERIFYING
            logger.info(f'After handling missing values in x_test:{self.x_test.columns}:\n:{self.x_test.shape}:\n:{self.x_test.isnull().sum()}')#VERIFYING
            logger.info(f'---------------------------------^missing values part^----------------------------------------')
        except Exception as e:                       #EXCEPTIONAL(ERROR HANDLING METHOD)
            err_type,err_line,err_msg =sys.exc_info()
            logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")



    def data_seperation(self):
        try:
            logger.info(f'-----------------------------------data separation part----------------------------------------------')
            self.x_train_num_cols = self.x_train.select_dtypes(exclude='object') #CREATING NEW COLUMN WHICH HAS ALL 'NUMERIC' COLUMNS
            self.x_test_num_cols = self.x_test.select_dtypes(exclude='object')#CREATING NEW COLUMN WHICH HAS ALL 'NUMERIC' COLUMNS
            logger.info(f'x_train_num_cols names: {self.x_train_num_cols.columns}: and size: {self.x_train_num_cols.shape}') #VERIFYING
            logger.info(f'x_test_num_cols names: {self.x_test_num_cols.columns}: and size: {self.x_test_num_cols.shape}') #VERIFYING
            self.x_train_cat_cols = self.x_train.select_dtypes(include='object')#CREATING NEW COLUMN WHICH HAS ALL 'CATEGORICAL' COLUMNS
            self.x_test_cat_cols = self.x_test.select_dtypes(include='object')#CREATING NEW COLUMN WHICH HAS ALL 'CATEGORICAL' COLUMNS
            logger.info(f'x_train_cat_cols names: {self.x_train_cat_cols.columns}: and size: {self.x_train_cat_cols.shape}') #VERIFYING
            logger.info(f'x_test_cat_cols names: {self.x_test_cat_cols.columns}: and size: {self.x_test_cat_cols.shape}') #VERIFYING
            logger.info(f'-------------------------------^data separation part^----------------------------------------')
        except Exception as e:  # EXCEPTIONAL(ERROR HANDLING METHOD)
            err_type, err_line, err_msg = sys.exc_info()
            logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")

    def variable_transformation(self):
        try:
            logger.info(f'-----------------------------variable transformation part------------------------------------')
            logger.info(f'Before x_train columns names : {self.x_train_num_cols.columns}') #CHECKING
            logger.info(f'Before x_test columns names : {self.x_test_num_cols.columns}') #CHECKING
            self.x_train_num_cols,self.x_test_num_cols = vt_outliers(self.x_train_num_cols,self.x_test_num_cols)
            logger.info(f'After x_train columns names : {self.x_train_num_cols.columns}') #VERIFYING
            logger.info(f'After x_test columns names : {self.x_test_num_cols.columns}')  #VERIFYING
            logger.info(f'-----------------------------variable transformation part------------------------------------')
        except Exception as e:  # EXCEPTIONAL(ERROR HANDLING METHOD
            err_type, err_line, err_msg = sys.exc_info()
            logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")

    def feature_selection(self):
        try:
            logger.info(f'--------------------------------feature selection part---------------------------------------')
            logger.info(f'x_train_num columns: {self.x_train_num_cols.shape}:\n: {self.x_train_num_cols.columns}')  # CHECKING
            logger.info(f'x_test_num columns: {self.x_test_num_cols.shape}:\n:{self.x_test_num_cols.columns}')  # CHECKING
            self.x_train_num_cols,self.x_test_num_cols= fm(self.x_train_num_cols,self.x_test_num_cols,self.y_train,self.y_test)#MODEL
            logger.info(f'After HYPOTHESIS TESTING train columns : {self.x_train_num_cols.shape}:\n:{self.x_train_num_cols.columns}')  # VERIFYING
            logger.info(f'After HYPOTHESIS TESTING test columns : {self.x_test_num_cols.shape}:\n:{self.x_test_num_cols.columns}')  # VERIFYING
            logger.info(f'--------------------------------feature selection part---------------------------------------')
        except Exception as e:  # EXCEPTIONAL(ERROR HANDLING METHOD)
            err_type, err_line, err_msg = sys.exc_info()
            logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")

    def cat_to_num(self):
        try:   #TRY BLOCK
            logger.info(f'--------------------------------cat to num part-----------------------------------------------')
            self.x_train_cat_cols,self.x_test_cat_cols = c_t_n(self.x_train_cat_cols,self.x_test_cat_cols) #MODEL
            #COMBINING DATA ALL NUMERICAL AND CATEGORICAL
            logger.info(f'--------------------------------combaining data part-----------------------------------------------')
            self.x_train_num_cols.reset_index(drop=True,inplace=True) #RESETING DATA
            self.x_train_cat_cols.reset_index(drop=True, inplace=True) #RESETING DATA
            self.x_test_num_cols.reset_index(drop=True, inplace=True) #RESETING DATA
            self.x_test_cat_cols.reset_index(drop=True, inplace=True) #RESETING DATA
            self.training_data = pd.concat([self.x_train_num_cols,self.x_train_cat_cols],axis=1) #CONCATNING BOTH NUM AND CAT COLS OF X_TRAIN INTO NEW COLUMN
            self.testing_data = pd.concat([self.x_test_num_cols, self.x_test_cat_cols], axis=1) #CONCATNING BOTH NUM AND CAT COLS OF X_TRAIN INTO NEW COLUMN
            logger.info(f'final training data : {self.training_data.shape}:\n:{self.training_data.columns}') #VERIFYING
            logger.info(f'training data null values : {self.training_data.isnull().sum()}') #VERIFYING
            logger.info(f'final testing data : {self.testing_data.shape}:\n: {self.testing_data.columns}') #VERIFYING
            logger.info(f'testing data null values : {self.testing_data.isnull().sum()}') #VERIFYING
            logger.info(f'--------------------------------combaining data part-----------------------------------------------')
            logger.info(f'--------------------------------cat to num part-----------------------------------------------')
        except Exception as e:  # EXCEPTIONAL(ERROR HANDLING METHOD)
            err_type, err_line, err_msg = sys.exc_info()
            logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")

    def data_balancing(self):  #CREATING FUNCTION
        try:  #TRY BLOCK
            logger.info(f'--------------------------------data balancing part-----------------------------------------------')
            logger.info(f'no.of row for good customer y_train:{1}:{sum(self.y_train==1)}') #CHECKING
            logger.info(f'no.of row for bad customer y_train:{0}:{sum(self.y_train==0)}') #CHECKING
            logger.info(f'Training data size : {self.training_data.shape}') #CHECKING
            sm = SMOTE(random_state=42)
            self.training_data_bal, self.y_train_bal = sm.fit_resample(self.training_data, self.y_train)
            logger.info(f'Number of Rows for GOOD customer y_train {1} : {sum(self.y_train_bal == 1)}')#VERIFYING
            logger.info(f'Number of Rows for BAD customer y_train {0} : {sum(self.y_train_bal == 0)}')#VERIFYING
            logger.info(f'Training_data_bal ds: {self.training_data_bal.shape}')#VERIFYING
            fs(self.training_data_bal, self.y_train_bal, self.testing_data, self.y_test)
            logger.info(f'Traning data bal columns: {self.training_data_bal.columns}')
        except Exception as e:  # EXCEPTIONAL(ERROR HANDLING METHOD)
            err_type, err_line, err_msg = sys.exc_info()
            logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")


if __name__ == '__main__' :
    try:                   #WRITING CODE UNDER 'TRY' BLOCK
        obj = CHURN('WA_Fn-UseC_-Telco-Customer-Churn.csv') #CREATING OBJECT TO CLASS AND CALLING CLASS
        obj.missing_values()                            #CALLING FUNCTION
        obj.data_seperation()                           #CALLING FUNCTION
        obj.variable_transformation()                   #CALLING FUNCTION
        obj.feature_selection()                         #CALLING FUNCTION
        obj.cat_to_num()                                #CALLING FUNCTION
        obj.data_balancing()                            #CALLING FUNCTION
    except Exception as e:  #EXCEPTIONAL(ERROR HANDLING METHOD)
        err_type,err_line,err_msg = sys.exc_info()
        logger.info(f"ERROR TYPE: {err_type} : CAUSE : {err_msg} : IN LINE: {err_line}")