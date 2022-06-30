from random import randint
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,RandomTreesEmbedding
from sklearn.model_selection import train_test_split
from pandas.api.types import is_numeric_dtype,is_string_dtype
from IPython.display import display
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from fastai.imports import*
from fastai.tabular import *
from pandas_summary import DataFrameSummary
import time
from sklearn import metrics
import pandas as pd
import numpy as np
import math
import tensorflow as tfds


df = pd.read_csv('..energy_dataset.csv', low_memory=False)
def train_cats(df):
    for n,c in df.items():
        if is_string_dtype(c):
            df[n]= c.astype('category').cat.as_ordered()
def apply_cats(df,train):
    for n,c in df.items():
        if train[n].dtype == 'category':
            df[n] = pd.Categorical(c,categories=train[n].cat.categories,ordered=True)
def numericalize(df,col,name):
    if not is_numeric_dtype(col):
        df[name] =col.cat.codes +1
def add_datepart(df,dt_name,drop=True):
    dt_column = df[dt_name]
    column_dtype = dt_column.dtype
    attr=['time','generation biomass','generation fossil brown coal/lignite','generation fossil coal-derived gas','generation fossil gas','generation fossil hard coal','generation fossil oil','generation fossil oil shale','generation fossil peat','generation geothermal','generation hydro pumped storage aggregated','generation hydro pumped storage consumption','generation hydro run-of-river and poundage','generation hydro water reservoir','generation marine','generation nuclear','generation other','generation other renewable','generation solar','generation waste','generation wind offshore','generation wind onshore','forecast solar day ahead','forecast wind offshore eday ahead','forecast wind onshore day ahead','total load forecast','total load actual','price day ahead','price actual']
    for a in attr:
        df['Date' + a.capitalize()] = getattr(dt_column.dt,a) 
        df['Date'+'Elapsed'] = dt_column.astype(np.int64) // 10**9
    if drop:
        df.drop(dt_name,axis=1,inplace=True)       
def fix_missing(df,col,name):
    if is_numeric_dtype(col):
        if pd.isnull(col).sum():
            df[name+'_na'] = pd.isnull(col)
        df[name] = col.fillna(col.median())
def numericalize(df,col,name):
    if not is_numeric_dtype(col):
        df[name]=col.cat.codes +1

def proc_df(df,y_fld,nan_dict=None,is_train=True):

    df = df.copy()
    y = df[y_fld].values

    df.drop(y_fld,axis=1,inplace=True)

    if nan_dict is None:
        nan_dict={}

    for n,c in df.items():
        fix_missing(df,c,n,nan_dict,is_train)
        numericalize(df,c,n)
def fix_missing(df,col,name,nan_dict,is_train):
    if is_train:
        if is_numeric_dtype(col):
            if pd.isnull(col).sum():
                df[name+'_na'] = pd.isnull(col)
                nan_dict[name] = col.median()
                df[name] = col.fillna(nan_dict[name])
    else:
        if is_numeric_dtype(col):
            if name in nan_dict:
                df[name+'_na'] = pd.isnull(col)
                df[name] = col.fillna(nan_dict[name])

            else:
                df[name]=col.fillna(df[name].median())
def split_train_val(df,n):
    return df[:n].copy(),df[n:].copy()

def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    print(f'RMSE of train set{rmse(m.predict(x_train),y_train)}')
    print(f'RMSE of validation set{rmse(m.predict(x_valid),y_valid)}')
    print(f'R^2 of train set{(m.score(x_train),y_train)}')
    print(f'R^2 of validation set{(m.score(x_valid),y_valid)}')

n_valid= 12000
n_train = len(df)-n_valid        
raw_train,raw_valid = split_train_val(df,n_train)

x_train,y_train,nas = proc_df(raw_train,'SalePrice')
x_valid,y_valid = proc_df(raw_valid,'SalePrice'nan_dict=nas,is_train=False)

x_train,y_train,nas = proc_df(raw_train,'SalePrice') #pathın yerinde raw_train vardı

x_valid,y_valid = proc_df(raw_valid,'SalePrice',nan_dict =nas)

def set_rf_samples(n):
    """Changes Scikit learn's random forests to give each tree a random sample of n random rows."""
    forest._generate_sample_indices = (lambda rs,n_samples:
        forest.check_random_state(rs).randint(0,n_samples,n))

def reset_rf_samples():
    """Undoes the changes produced by set_rf_samples."""

forest._generate_sample_indices = (lambda rs ,n_samples:
    forest._check_random_state(rs).randint(0,n_samples,n_samples))


set_rf_samples(30000)

m = RandomForestRegressor(n_estimators=10,n_jobs=-1)
%time m.fit(x_train,y_train)
print_score(m)

m = RandomForestRegressor(n_estimators=40,n_jobs=-1)
%time m.fit(x_train,y_train)
print_score(m)

m = RandomForestRegressor(n_estimators=100,n_jobs=-1)
%time m.fit(x_train,y_train)
print_score(m)
     

#en iyi skor 10 ağaçta çıkmıştır













